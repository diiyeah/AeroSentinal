"""
AeroSentinal — Synthetic Data Generation Runner
=================================================
Runs all synthetic data generators for subsystems that lack
real-world public datasets.

Usage:
  python scripts/generate_synthetic.py
  python scripts/generate_synthetic.py --subsystem landing_gear
  python scripts/generate_synthetic.py --subsystem apu
  python scripts/generate_synthetic.py --subsystem ecs
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"


def generate_landing_gear():
    """Generate landing gear / brake synthetic dataset."""
    from ml.landing_gear.synthetic_generator import generate_landing_gear_dataset
    output_path = DATA_DIR / "landing_gear" / "landing_gear_synthetic.csv"
    df = generate_landing_gear_dataset(n_samples=5000, save_path=output_path)
    return df


def generate_apu():
    """Generate APU synthetic dataset."""
    from ml.apu.synthetic_generator import generate_apu_dataset
    output_path = DATA_DIR / "apu" / "apu_synthetic.csv"
    df = generate_apu_dataset(n_units=200, save_path=output_path)
    return df


def generate_ecs():
    """Generate ECS scenario dataset."""
    from ml.ecs.simulator import ECSSimulator
    output_path = DATA_DIR / "ecs" / "ecs_scenarios.csv"
    sim = ECSSimulator()
    df = sim.generate_dataset(n_scenarios=500, save_path=output_path)
    return df


GENERATORS = {
    "landing_gear": generate_landing_gear,
    "apu": generate_apu,
    "ecs": generate_ecs,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for AeroSentinal"
    )
    parser.add_argument(
        "--subsystem",
        choices=list(GENERATORS.keys()) + ["all"],
        default="all",
        help="Which subsystem to generate data for (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AeroSentinal - Synthetic Data Generation")
    print("  [!] All generated data is clearly labeled as SYNTHETIC")
    print("=" * 60)

    subsystems = list(GENERATORS.keys()) if args.subsystem == "all" else [args.subsystem]

    results = {}
    for name in subsystems:
        print(f"\n{'='*60}")
        print(f"  Generating: {name}")
        print(f"{'='*60}")
        try:
            df = GENERATORS[name]()
            results[name] = True
            print(f"  [OK] {name}: {df.shape[0]} records generated")
        except Exception as e:
            results[name] = False
            print(f"  [X] {name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("  Generation Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "[OK] Generated" if success else "[X] Failed"
        print(f"  {status}: {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
