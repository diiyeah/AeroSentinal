"""
AeroSentinal — APU Synthetic Data Generator
============================================
Physics-informed synthetic dataset modeling APU health/degradation.

SYNTHETIC DATA NOTICE:
  This generator produces SYNTHETIC data. No real-world APU telemetry
  dataset was used. Every record carries is_synthetic_data=True.

Physics Model:
  - EGT startup curve: exponential rise to steady-state
    T(t) = T_ambient + (T_ss - T_ambient) × (1 - exp(-t / τ))
  - Fouling parameter (0-100%) affects:
    • Steady-state EGT (increases with fouling)
    • Time constant τ (increases with fouling — slower startup)
    • Fuel flow rate (increases with fouling — less efficient)
  - Fleet baseline: healthy APU parameters define the reference
  - Health score: deviation from fleet-nominal baseline

Health Classes:
  0: healthy        — within fleet-nominal envelope
  1: degraded_mild  — slight performance deviation
  2: degraded_moderate — noticeable deviation, monitor closely
  3: degraded_severe   — approaching limits, schedule maintenance
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# APU Physical Parameters (Typical small gas turbine)
# ============================================================

# Healthy baseline parameters
HEALTHY_PARAMS = {
    "egt_steady_state_c": 620.0,    # Healthy EGT at steady state (°C)
    "startup_tau_s": 8.0,           # Time constant for startup (seconds)
    "fuel_flow_kg_h": 120.0,        # Fuel flow at steady state (kg/hr)
    "turbine_inlet_temp_c": 850.0,  # Turbine inlet temperature (°C)
    "rpm_steady": 99.5,             # Steady-state RPM (% of rated)
}

T_AMBIENT = 25.0  # Ambient temperature (°C)

# Fouling effects (per percentage point of fouling)
FOULING_EFFECTS = {
    "egt_increase_per_pct": 1.8,      # °C per 1% fouling
    "tau_increase_per_pct": 0.15,     # seconds per 1% fouling
    "fuel_flow_increase_per_pct": 0.8, # kg/hr per 1% fouling
    "tit_increase_per_pct": 1.2,      # °C per 1% fouling (turbine inlet)
    "rpm_decrease_per_pct": 0.03,     # % RPM per 1% fouling
}

# Health class thresholds (based on fouling %)
HEALTH_THRESHOLDS = {
    "healthy":            (0, 15),
    "degraded_mild":      (15, 35),
    "degraded_moderate":  (35, 60),
    "degraded_severe":    (60, 100),
}

HEALTH_LABELS = {
    "healthy": 0,
    "degraded_mild": 1,
    "degraded_moderate": 2,
    "degraded_severe": 3,
}


# ============================================================
# EGT Startup Curve Simulation
# ============================================================

def simulate_egt_startup(fouling_pct: float,
                         rng: np.random.Generator,
                         n_timesteps: int = 60) -> dict:
    """
    Simulate a single APU startup with given fouling level.
    Returns sensor readings at steady-state plus curve characteristics.
    """
    # Apply fouling effects to baseline parameters
    egt_ss = (HEALTHY_PARAMS["egt_steady_state_c"]
              + fouling_pct * FOULING_EFFECTS["egt_increase_per_pct"])
    tau = (HEALTHY_PARAMS["startup_tau_s"]
           + fouling_pct * FOULING_EFFECTS["tau_increase_per_pct"])
    fuel_flow = (HEALTHY_PARAMS["fuel_flow_kg_h"]
                 + fouling_pct * FOULING_EFFECTS["fuel_flow_increase_per_pct"])
    tit = (HEALTHY_PARAMS["turbine_inlet_temp_c"]
           + fouling_pct * FOULING_EFFECTS["tit_increase_per_pct"])
    rpm = (HEALTHY_PARAMS["rpm_steady"]
           - fouling_pct * FOULING_EFFECTS["rpm_decrease_per_pct"])

    # Add sensor noise
    noise_scale = 0.02  # 2% noise
    egt_ss_noisy = egt_ss * (1 + rng.normal(0, noise_scale))
    tau_noisy = max(1.0, tau * (1 + rng.normal(0, noise_scale)))
    fuel_flow_noisy = fuel_flow * (1 + rng.normal(0, noise_scale))
    tit_noisy = tit * (1 + rng.normal(0, noise_scale))
    rpm_noisy = rpm * (1 + rng.normal(0, noise_scale * 0.5))

    # Simulate startup curve
    time = np.arange(n_timesteps)
    egt_curve = T_AMBIENT + (egt_ss_noisy - T_AMBIENT) * (1 - np.exp(-time / tau_noisy))

    # Extract curve features
    # Time to reach 90% of steady-state EGT
    target_90 = T_AMBIENT + 0.9 * (egt_ss_noisy - T_AMBIENT)
    time_to_90 = -tau_noisy * np.log(1 - 0.9) if tau_noisy > 0 else 0
    
    # Maximum EGT rate of change (at t=0)
    max_egt_rate = (egt_ss_noisy - T_AMBIENT) / tau_noisy if tau_noisy > 0 else 0

    # EGT margin (how close to redline of 700°C for healthy, or absolute 800°C)
    egt_margin = 800.0 - egt_ss_noisy

    return {
        "egt_steady_state_c": round(egt_ss_noisy, 1),
        "startup_time_constant_s": round(tau_noisy, 2),
        "fuel_flow_kg_h": round(fuel_flow_noisy, 1),
        "turbine_inlet_temp_c": round(tit_noisy, 1),
        "rpm_pct": round(rpm_noisy, 2),
        "time_to_90pct_s": round(time_to_90, 1),
        "max_egt_rate_c_per_s": round(max_egt_rate, 2),
        "egt_margin_c": round(egt_margin, 1),
        "egt_curve_sample": egt_curve[::5].tolist(),  # Sample every 5th point
    }


# ============================================================
# Dataset Generation
# ============================================================

def generate_apu_dataset(
    n_units: int = 200,
    random_seed: int = 42,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a fleet of synthetic APU units with varying degradation.

    Each record represents one APU unit's current health state,
    characterized by steady-state sensor readings and startup behavior.

    Returns:
        DataFrame with sensor readings, fouling level, health class,
        and is_synthetic_data flag.
    """
    rng = np.random.default_rng(random_seed)

    # Generate fouling levels with controlled distribution
    # Use a mixture to ensure representation of all classes
    fouling_levels = np.concatenate([
        rng.uniform(0, 15, size=n_units // 4),       # healthy
        rng.uniform(15, 35, size=n_units // 4),       # mild
        rng.uniform(35, 60, size=n_units // 4),       # moderate
        rng.uniform(60, 100, size=n_units - 3 * (n_units // 4)),  # severe
    ])
    rng.shuffle(fouling_levels)

    records = []
    for i, fouling_pct in enumerate(fouling_levels):
        # Simulate startup
        readings = simulate_egt_startup(fouling_pct, rng)

        # Determine health class
        health_label = "healthy"
        for label, (low, high) in HEALTH_THRESHOLDS.items():
            if low <= fouling_pct < high:
                health_label = label
                break

        # Compute health score (0 = worst, 100 = best)
        health_score = max(0, 100 - fouling_pct)

        # Remove the curve sample for CSV (it's a list)
        curve_sample = readings.pop("egt_curve_sample")

        records.append({
            "unit_id": f"APU-{i+1:03d}",
            "fouling_pct": round(fouling_pct, 1),
            **readings,
            "health_score": round(health_score, 1),
            "health_class": HEALTH_LABELS[health_label],
            "health_label": health_label,
            "is_synthetic_data": True,
        })

    df = pd.DataFrame(records)

    # Report distribution
    print("\n  APU Synthetic Dataset — Health Distribution:")
    print("  " + "-" * 50)
    for label, cls in HEALTH_LABELS.items():
        count = (df["health_class"] == cls).sum()
        pct = 100 * count / len(df)
        print(f"    {cls} ({label:20s}): {count:4d} ({pct:5.1f}%)")

    # Save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n  [OK] Saved {len(df)} APU unit records to {save_path}")

    return df


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "processed" / "apu" / "apu_synthetic.csv"

    print("=" * 60)
    print("  AeroSentinal - APU Synthetic Data Generator")
    print("  [!] SYNTHETIC DATA: Physics-informed, not real telemetry")
    print("=" * 60)

    df = generate_apu_dataset(n_units=200, save_path=output_path)
    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Fouling range: {df['fouling_pct'].min():.1f}% - {df['fouling_pct'].max():.1f}%")
    print(f"  EGT range: {df['egt_steady_state_c'].min():.0f}°C - {df['egt_steady_state_c'].max():.0f}°C")
