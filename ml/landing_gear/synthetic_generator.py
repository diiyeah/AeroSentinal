"""
AeroSentinal — Landing Gear / Brakes Synthetic Data Generator
==============================================================
Physics-informed synthetic dataset modeling brake wear severity.

SYNTHETIC DATA NOTICE:
  This generator produces SYNTHETIC data based on simplified physics models.
  No real-world landing gear telemetry dataset was used.
  Every record carries is_synthetic_data=True.
  This is clearly labeled in all API responses and UI displays.

Physics Model:
  - Brake energy: E_brake = ½ × m × v² × braking_fraction
  - Temperature rise: ΔT = E_brake / (m_brake × c_p)
  - Wear accumulation: modeled as cumulative thermal cycles with
    nonlinear acceleration at high temperatures
  - Severity classes based on cumulative wear index

Classes:
  0: nominal       — new or recently serviced brakes
  1: light_wear    — within normal service limits
  2: moderate_wear — approaching service limit
  3: heavy_wear    — at service limit, schedule replacement
  4: critical      — beyond service limit, immediate action required
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Physical constants and parameters
# ============================================================

# Brake material properties (carbon-carbon composite, typical)
BRAKE_MASS_KG = 25.0          # per-wheel brake assembly mass
SPECIFIC_HEAT = 1200.0         # J/(kg·K) for carbon-carbon
MAX_BRAKE_TEMP_C = 900.0       # absolute max observed temperature
AMBIENT_TEMP_C = 25.0          # ground ambient baseline

# Aircraft parameters
MASS_RANGE_KG = (50_000, 80_000)   # operating empty + payload
SPEED_RANGE_KTS = (120, 160)        # typical touchdown speed
BRAKE_PRESSURE_RANGE_PSI = (500, 3000)

# Runway surface friction coefficients
RUNWAY_CONDITIONS = {
    "dry":  0.8,
    "wet":  0.5,
    "icy":  0.2,
}

# Wear severity thresholds (cumulative wear index)
WEAR_THRESHOLDS = {
    "nominal":       (0.0, 0.15),
    "light_wear":    (0.15, 0.40),
    "moderate_wear": (0.40, 0.65),
    "heavy_wear":    (0.65, 0.85),
    "critical":      (0.85, 1.0),
}

CLASS_LABELS = {
    "nominal": 0,
    "light_wear": 1,
    "moderate_wear": 2,
    "heavy_wear": 3,
    "critical": 4,
}

# ============================================================
# Physics simulation
# ============================================================

def compute_brake_energy(aircraft_mass_kg: float,
                         touchdown_speed_kts: float,
                         braking_fraction: float) -> float:
    """
    Compute kinetic energy absorbed by brakes.
    braking_fraction: fraction of deceleration from brakes vs thrust reversers/drag
    """
    speed_ms = touchdown_speed_kts * 0.5144  # knots to m/s
    total_ke = 0.5 * aircraft_mass_kg * speed_ms ** 2
    return total_ke * braking_fraction


def compute_temp_rise(brake_energy_j: float) -> float:
    """Temperature rise from absorbed energy (simplified lumped model)."""
    return brake_energy_j / (BRAKE_MASS_KG * SPECIFIC_HEAT)


def compute_wear_index(n_cycles: int,
                       peak_temps: np.ndarray,
                       brake_pressures: np.ndarray) -> float:
    """
    Cumulative wear index (0-1) based on thermal cycling history.
    High temperatures accelerate wear nonlinearly.
    """
    # Normalized temperature contribution (nonlinear above 500°C)
    temp_factor = np.where(
        peak_temps > 500,
        (peak_temps / MAX_BRAKE_TEMP_C) ** 2.5,
        (peak_temps / MAX_BRAKE_TEMP_C) ** 1.5,
    )

    # Pressure contribution (normalized)
    pressure_factor = brake_pressures / BRAKE_PRESSURE_RANGE_PSI[1]

    # Cumulative wear with cycle count normalization
    raw_wear = np.sum(temp_factor * pressure_factor) / max(n_cycles, 1)
    cycle_factor = np.log1p(n_cycles) / np.log1p(2000)  # normalized to ~2000 cycle life

    return float(np.clip(raw_wear * cycle_factor, 0.0, 1.0))


# ============================================================
# Dataset generation
# ============================================================

def generate_landing_gear_dataset(
    n_samples: int = 5000,
    random_seed: int = 42,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a physics-informed synthetic landing gear / brake dataset.

    Each record represents a single landing event with sensor readings
    and a wear severity classification.

    Returns:
        DataFrame with columns:
          - aircraft_mass_kg, touchdown_speed_kts, brake_pressure_psi
          - runway_friction_coeff, braking_fraction
          - brake_energy_mj, peak_brake_temp_c, temp_rise_c
          - cumulative_cycles, wear_index
          - severity_class (0-4), severity_label
          - is_synthetic_data (always True)
    """
    rng = np.random.default_rng(random_seed)

    records = []

    for i in range(n_samples):
        # Sample aircraft parameters
        aircraft_mass = rng.uniform(*MASS_RANGE_KG)
        touchdown_speed = rng.uniform(*SPEED_RANGE_KTS)
        brake_pressure = rng.uniform(*BRAKE_PRESSURE_RANGE_PSI)

        # Runway condition (weighted: 70% dry, 20% wet, 10% icy)
        condition = rng.choice(
            list(RUNWAY_CONDITIONS.keys()),
            p=[0.70, 0.20, 0.10]
        )
        friction_coeff = RUNWAY_CONDITIONS[condition]

        # Braking fraction depends on thrust reverser availability and conditions
        # Higher friction = more brake authority, but also more TR effectiveness
        base_braking_frac = rng.uniform(0.4, 0.8)
        braking_fraction = base_braking_frac * (0.5 + 0.5 * friction_coeff)

        # Physics computations
        brake_energy = compute_brake_energy(aircraft_mass, touchdown_speed, braking_fraction)
        temp_rise = compute_temp_rise(brake_energy)
        peak_temp = AMBIENT_TEMP_C + temp_rise

        # Cumulative cycle history (simulates brake life stage)
        # Use a distribution that creates desired class balance
        # Controlled to avoid extreme imbalance (target ~4:1 max)
        cycle_stage = rng.beta(2.0, 2.0)  # centered distribution
        cumulative_cycles = int(cycle_stage * 2000)

        # Generate synthetic history for wear calculation
        n_history = min(cumulative_cycles, 50)  # use last 50 cycles
        hist_temps = rng.normal(peak_temp, peak_temp * 0.1, size=max(n_history, 1))
        hist_pressures = rng.normal(brake_pressure, brake_pressure * 0.05, size=max(n_history, 1))

        wear_index = compute_wear_index(cumulative_cycles, hist_temps, hist_pressures)

        # Classify severity
        severity_label = "nominal"
        for label, (low, high) in WEAR_THRESHOLDS.items():
            if low <= wear_index < high:
                severity_label = label
                break
        if wear_index >= 1.0:
            severity_label = "critical"

        severity_class = CLASS_LABELS[severity_label]

        records.append({
            "aircraft_mass_kg": round(aircraft_mass, 1),
            "touchdown_speed_kts": round(touchdown_speed, 1),
            "brake_pressure_psi": round(brake_pressure, 1),
            "runway_friction_coeff": round(friction_coeff, 2),
            "braking_fraction": round(braking_fraction, 3),
            "brake_energy_mj": round(brake_energy / 1e6, 3),
            "peak_brake_temp_c": round(min(peak_temp, MAX_BRAKE_TEMP_C), 1),
            "temp_rise_c": round(temp_rise, 1),
            "cumulative_cycles": cumulative_cycles,
            "wear_index": round(wear_index, 4),
            "severity_class": severity_class,
            "severity_label": severity_label,
            "is_synthetic_data": True,
        })

    df = pd.DataFrame(records)

    # Report class distribution
    print("\n  Landing Gear Synthetic Dataset — Class Distribution:")
    print("  " + "-" * 45)
    for label, cls in CLASS_LABELS.items():
        count = (df["severity_class"] == cls).sum()
        pct = 100 * count / len(df)
        print(f"    {cls} ({label:15s}): {count:5d} ({pct:5.1f}%)")

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n  [OK] Saved {len(df)} records to {save_path}")

    return df


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "ml" / "data" / "processed" / "landing_gear" / "landing_gear_synthetic.csv"
    
    print("=" * 60)
    print("  AeroSentinal - Landing Gear Synthetic Data Generator")
    print("  [!] SYNTHETIC DATA: Physics-informed, not real telemetry")
    print("=" * 60)
    
    df = generate_landing_gear_dataset(n_samples=5000, save_path=output_path)
    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
