"""
AeroSentinal — ECS Reverse-Brayton-Cycle Thermodynamic Simulator
=================================================================
Parametrized Environmental Control System simulator with controllable
heat exchanger fouling and cross-domain engine coupling.

SYNTHETIC DATA NOTICE:
  This simulator produces SYNTHETIC data based on thermodynamic first
  principles. No real ECS telemetry was used. All outputs carry
  is_synthetic_data=True.

Physics Model (Reverse Brayton / Air Cycle Machine):
  - Bleed air enters at T1/P1 from engine compressor
  - Primary Heat Exchanger (PHX): cools bleed air using ram air
  - Air Cycle Machine (ACM) Compressor: compresses the air
  - Secondary Heat Exchanger (SHX): further cooling with ram air
  - ACM Turbine: expands air, producing cold air
  - Water Separator: removes moisture
  - Mixed air delivered to cabin at T_cabin target

  All computations use ideal gas relations with constant specific heat:
    T2/T1 = (P2/P1)^((γ-1)/γ)  for isentropic processes
    Q = ṁ × c_p × ΔT           for heat exchange

Cross-Domain Coupling (the differentiator):
  When ECS heat exchanger fouling increases:
  1. PHX/SHX effectiveness drops → less cooling
  2. Cabin temperature rises above setpoint
  3. ECS controller compensates by requesting MORE bleed air
  4. Increased bleed extraction raises engine compressor discharge temp
  5. Engine-only monitoring incorrectly flags "compressor degradation"

  The attribution module (ml/ecs/attribution.py) determines whether a
  temperature anomaly is engine-original or ECS-induced.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Thermodynamic Constants
# ============================================================

GAMMA = 1.4                # Ratio of specific heats (air)
CP_AIR = 1005.0            # Specific heat at constant pressure (J/kg·K)
R_AIR = 287.0              # Gas constant for air (J/kg·K)
GAMMA_RATIO = (GAMMA - 1) / GAMMA  # 0.2857...


# ============================================================
# ECS System Configuration
# ============================================================

@dataclass
class ECSConfig:
    """Configuration for one ECS pack."""

    # Bleed air source conditions (from engine compressor stage 8-10)
    bleed_temp_k: float = 473.15       # ~200°C bleed air temperature
    bleed_pressure_kpa: float = 350.0   # Bleed air pressure

    # Ram air conditions (at altitude)
    ram_temp_k: float = 223.15         # ~-50°C at FL350
    ram_pressure_kpa: float = 24.0     # Ambient at FL350

    # Heat exchanger nominal effectiveness (0-1)
    phx_effectiveness: float = 0.85     # Primary HX
    shx_effectiveness: float = 0.80     # Secondary HX

    # ACM parameters
    acm_compressor_pr: float = 2.5      # ACM compressor pressure ratio
    acm_turbine_pr: float = 2.0         # ACM turbine expansion ratio
    acm_compressor_eta: float = 0.82    # Isentropic efficiency
    acm_turbine_eta: float = 0.85       # Isentropic efficiency

    # Cabin requirements
    cabin_target_temp_k: float = 297.15  # 24°C cabin target
    cabin_flow_req_kg_s: float = 0.5     # Baseline cabin air mass flow

    # Water separator
    ws_pressure_drop_pct: float = 3.0    # Pressure drop across water separator


@dataclass
class ECSState:
    """State variables at each ECS stage."""
    # Stage temperatures (K) and pressures (kPa)
    T1: float = 0.0  # Bleed air inlet
    P1: float = 0.0

    T2: float = 0.0  # After Primary HX
    P2: float = 0.0

    T3: float = 0.0  # After ACM Compressor
    P3: float = 0.0

    T4: float = 0.0  # After Secondary HX
    P4: float = 0.0

    T5: float = 0.0  # After ACM Turbine
    P5: float = 0.0

    T6: float = 0.0  # After Water Separator (cabin supply)
    P6: float = 0.0

    # Derived quantities
    mass_flow_kg_s: float = 0.0
    cooling_capacity_kw: float = 0.0
    cabin_temp_achieved_k: float = 0.0
    cop: float = 0.0  # Coefficient of performance

    # Fouling info
    fouling_pct: float = 0.0
    phx_effectiveness_actual: float = 0.0
    shx_effectiveness_actual: float = 0.0

    # Cross-domain coupling
    bleed_air_demand_increase_pct: float = 0.0
    engine_temp_perturbation_k: float = 0.0


# ============================================================
# Core Simulator
# ============================================================

class ECSSimulator:
    """
    Reverse-Brayton-cycle ECS simulator with fouling and coupling effects.
    """

    def __init__(self, config: Optional[ECSConfig] = None):
        self.config = config or ECSConfig()

    def simulate(self,
                 fouling_pct: float = 0.0,
                 flight_phase: str = "cruise",
                 ambient_temp_offset_k: float = 0.0) -> ECSState:
        """
        Run one steady-state ECS simulation.

        Args:
            fouling_pct: Heat exchanger fouling level (0-100%).
                         Reduces HX effectiveness linearly.
            flight_phase: "ground", "climb", "cruise", "descent"
            ambient_temp_offset_k: Offset to ram air temperature (e.g., hot day)

        Returns:
            ECSState with all stage conditions and derived quantities.
        """
        cfg = self.config
        state = ECSState()
        state.fouling_pct = fouling_pct

        # Adjust ram air for flight phase
        phase_adjustments = {
            "ground":  {"ram_t": 288.15 + ambient_temp_offset_k, "ram_p": 101.3},
            "climb":   {"ram_t": 253.15 + ambient_temp_offset_k, "ram_p": 60.0},
            "cruise":  {"ram_t": cfg.ram_temp_k + ambient_temp_offset_k, "ram_p": cfg.ram_pressure_kpa},
            "descent": {"ram_t": 263.15 + ambient_temp_offset_k, "ram_p": 55.0},
        }
        phase = phase_adjustments.get(flight_phase, phase_adjustments["cruise"])
        ram_temp = phase["ram_t"]
        ram_pressure = phase["ram_p"]

        # === Stage 1: Bleed air inlet ===
        state.T1 = cfg.bleed_temp_k
        state.P1 = cfg.bleed_pressure_kpa

        # === Apply fouling to heat exchangers ===
        # Fouling linearly reduces effectiveness
        fouling_factor = 1.0 - (fouling_pct / 100.0) * 0.6  # Max 60% reduction at 100% fouling
        state.phx_effectiveness_actual = cfg.phx_effectiveness * fouling_factor
        state.shx_effectiveness_actual = cfg.shx_effectiveness * fouling_factor

        # === Stage 2: After Primary Heat Exchanger ===
        # PHX cools bleed air toward ram air temperature
        # Q = ε × ṁ × cp × (T_hot_in - T_cold_in)
        T2_ideal = ram_temp  # If perfect HX, bleed air cools to ram temp
        state.T2 = state.T1 - state.phx_effectiveness_actual * (state.T1 - T2_ideal)
        state.P2 = state.P1 * 0.97  # Small pressure drop across HX

        # === Stage 3: After ACM Compressor ===
        # Isentropic compression: T3s/T2 = PR^((γ-1)/γ)
        T3_isentropic = state.T2 * (cfg.acm_compressor_pr ** GAMMA_RATIO)
        # Actual temperature with efficiency losses
        state.T3 = state.T2 + (T3_isentropic - state.T2) / cfg.acm_compressor_eta
        state.P3 = state.P2 * cfg.acm_compressor_pr

        # === Stage 4: After Secondary Heat Exchanger ===
        T4_ideal = ram_temp
        state.T4 = state.T3 - state.shx_effectiveness_actual * (state.T3 - T4_ideal)
        state.P4 = state.P3 * 0.96  # Pressure drop

        # === Stage 5: After ACM Turbine ===
        # Isentropic expansion
        T5_isentropic = state.T4 / (cfg.acm_turbine_pr ** GAMMA_RATIO)
        # Actual with efficiency
        state.T5 = state.T4 - cfg.acm_turbine_eta * (state.T4 - T5_isentropic)
        state.P5 = state.P4 / cfg.acm_turbine_pr

        # === Stage 6: After Water Separator ===
        # Small pressure drop, temperature essentially unchanged
        state.T6 = state.T5  # Negligible temp change
        state.P6 = state.P5 * (1 - cfg.ws_pressure_drop_pct / 100)

        # === Cabin temperature achieved ===
        # Mix ECS output with recirculated cabin air (50/50 typical)
        recirc_temp = cfg.cabin_target_temp_k
        state.cabin_temp_achieved_k = 0.5 * state.T6 + 0.5 * recirc_temp

        # === Mass flow calculation ===
        # Controller adjusts mass flow to achieve cabin target
        # If cooling is insufficient (fouled HX), controller demands more flow
        temp_deficit = state.cabin_temp_achieved_k - cfg.cabin_target_temp_k
        if temp_deficit > 0 and state.T6 < cfg.cabin_target_temp_k:
            # Need more flow to compensate for reduced cooling capacity
            flow_multiplier = 1.0 + (temp_deficit / 10.0)  # 10K deficit → 2x flow
        elif temp_deficit > 0:
            # Cooling output is actually warm — need much more flow
            flow_multiplier = 1.0 + (fouling_pct / 100.0) * 1.5
        else:
            flow_multiplier = 1.0

        state.mass_flow_kg_s = cfg.cabin_flow_req_kg_s * flow_multiplier

        # === Cooling capacity ===
        state.cooling_capacity_kw = (
            state.mass_flow_kg_s * CP_AIR * (cfg.cabin_target_temp_k - state.T6) / 1000
        )

        # === COP ===
        work_input = state.mass_flow_kg_s * CP_AIR * (state.T3 - state.T2)  # Compressor work
        if work_input > 0:
            state.cop = abs(state.cooling_capacity_kw * 1000) / work_input
        else:
            state.cop = 0.0

        # === CROSS-DOMAIN COUPLING EFFECT ===
        # More bleed air demand → higher engine workload
        bleed_increase = (flow_multiplier - 1.0) * 100  # percentage increase
        state.bleed_air_demand_increase_pct = round(bleed_increase, 1)

        # Engine temperature perturbation due to excess bleed extraction
        # Rule of thumb: 1% more bleed ≈ 2-3K compressor discharge temp increase
        # (due to reduced compressor surge margin and operating point shift)
        state.engine_temp_perturbation_k = round(bleed_increase * 2.5, 1)

        return state

    def generate_dataset(self,
                         n_scenarios: int = 500,
                         random_seed: int = 42,
                         save_path: Optional[str | Path] = None) -> pd.DataFrame:
        """
        Generate a dataset of ECS operating scenarios with varying fouling.

        Returns DataFrame with all ECS state variables and coupling effects.
        """
        rng = np.random.default_rng(random_seed)

        records = []
        flight_phases = ["ground", "climb", "cruise", "descent"]

        for i in range(n_scenarios):
            # Vary fouling
            fouling = rng.uniform(0, 100)

            # Vary flight phase
            phase = rng.choice(flight_phases, p=[0.1, 0.15, 0.6, 0.15])

            # Vary ambient temperature
            ambient_offset = rng.normal(0, 5)  # ±5K variation

            state = self.simulate(
                fouling_pct=fouling,
                flight_phase=phase,
                ambient_temp_offset_k=ambient_offset,
            )

            records.append({
                "scenario_id": i + 1,
                "flight_phase": phase,
                "ambient_temp_offset_k": round(ambient_offset, 1),
                "fouling_pct": round(fouling, 1),
                "T1_bleed_k": round(state.T1, 1),
                "T2_after_phx_k": round(state.T2, 1),
                "T3_after_acm_comp_k": round(state.T3, 1),
                "T4_after_shx_k": round(state.T4, 1),
                "T5_after_acm_turb_k": round(state.T5, 1),
                "T6_cabin_supply_k": round(state.T6, 1),
                "P1_kpa": round(state.P1, 1),
                "P6_kpa": round(state.P6, 1),
                "mass_flow_kg_s": round(state.mass_flow_kg_s, 3),
                "cooling_capacity_kw": round(state.cooling_capacity_kw, 2),
                "cop": round(state.cop, 3),
                "phx_effectiveness": round(state.phx_effectiveness_actual, 3),
                "shx_effectiveness": round(state.shx_effectiveness_actual, 3),
                "cabin_temp_achieved_k": round(state.cabin_temp_achieved_k, 1),
                "bleed_demand_increase_pct": state.bleed_air_demand_increase_pct,
                "engine_temp_perturbation_k": state.engine_temp_perturbation_k,
                "is_synthetic_data": True,
            })

        df = pd.DataFrame(records)

        # Report
        print("\n  ECS Synthetic Dataset Summary:")
        print("  " + "-" * 50)
        print(f"    Scenarios: {len(df)}")
        print(f"    Fouling range: {df['fouling_pct'].min():.0f}% - {df['fouling_pct'].max():.0f}%")
        print(f"    Engine temp perturbation range: "
              f"{df['engine_temp_perturbation_k'].min():.0f}K - "
              f"{df['engine_temp_perturbation_k'].max():.0f}K")
        print(f"    Bleed demand increase range: "
              f"{df['bleed_demand_increase_pct'].min():.0f}% - "
              f"{df['bleed_demand_increase_pct'].max():.0f}%")

        # Flight phase distribution
        print("\n    Flight Phase Distribution:")
        for phase in flight_phases:
            count = (df["flight_phase"] == phase).sum()
            print(f"      {phase:10s}: {count}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\n  [OK] Saved {len(df)} scenarios to {save_path}")

        return df


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "processed" / "ecs" / "ecs_scenarios.csv"

    print("=" * 60)
    print("  AeroSentinal - ECS Reverse-Brayton Cycle Simulator")
    print("  [!] SYNTHETIC DATA: Thermodynamic model, not real telemetry")
    print("=" * 60)

    sim = ECSSimulator()

    # Demo: show coupling effect at different fouling levels
    print("\n  Coupling Demonstration (cruise phase):")
    print("  " + "-" * 65)
    print(f"  {'Fouling':>8s} | {'Cabin T':>8s} | {'Bleed ↑':>8s} | {'Engine ΔT':>10s} | {'COP':>6s}")
    print("  " + "-" * 65)
    for fouling in [0, 10, 25, 40, 60, 80, 100]:
        state = sim.simulate(fouling_pct=fouling)
        cabin_c = state.cabin_temp_achieved_k - 273.15
        print(f"  {fouling:7d}% | {cabin_c:7.1f}°C | {state.bleed_air_demand_increase_pct:7.1f}% | "
              f"{state.engine_temp_perturbation_k:9.1f}K | {state.cop:6.3f}")

    # Generate full dataset
    print()
    df = sim.generate_dataset(n_scenarios=500, save_path=output_path)
