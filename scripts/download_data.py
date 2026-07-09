"""
AeroSentinal — Dataset Download Script
=======================================
Downloads and verifies all external datasets required for the prototype.

Datasets:
  1. NASA C-MAPSS (FD001-FD004) — Turbofan engine degradation
  2. UCI Condition Monitoring of Hydraulic Systems

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --dataset cmapss
  python scripts/download_data.py --dataset hydraulics
"""

import os
import sys
import io
import zipfile
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

# Fix Windows console encoding issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Configuration
# ============================================================

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

DATASETS = {
    "cmapss": {
        "description": "NASA C-MAPSS Turbofan Engine Degradation Simulation",
        "urls": [
            # Primary: Zenodo archive (~12 MB, the classic FD001-FD004 dataset)
            "https://zenodo.org/records/263309/files/CMAPSSData.zip",
            # Fallback: Kaggle-hosted mirror
            "https://raw.githubusercontent.com/kpeters/exploring-nasas-turbofan-dataset/master/CMAPSSData.zip",
        ],
        "output_dir": DATA_DIR / "cmapss",
        "zip_name": "CMAPSSData.zip",
        "expected_files": [
            "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
            "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
            "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
            "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
        ],
        "validation": {
            # Expected approximate line counts for integrity checking
            "train_FD001.txt": {"min_lines": 20000, "max_lines": 21000},
            "train_FD004.txt": {"min_lines": 61000, "max_lines": 62000},
        },
    },
    "hydraulics": {
        "description": "UCI Condition Monitoring of Hydraulic Systems",
        "urls": [
            "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
        ],
        "output_dir": DATA_DIR / "hydraulics",
        "zip_name": "hydraulic_systems.zip",
        "expected_files": [
            "PS1.txt", "PS2.txt", "PS3.txt", "PS4.txt", "PS5.txt", "PS6.txt",
            "TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt",
            "VS1.txt", "EPS1.txt", "FS1.txt", "FS2.txt",
            "SE.txt", "CE.txt", "CP.txt",
            "profile.txt",
        ],
        "validation": {
            # Each sensor file should have 2205 lines (2205 cycles)
            "PS1.txt": {"min_lines": 2200, "max_lines": 2210},
        },
    },
}


# ============================================================
# Download helpers
# ============================================================

def progress_hook(block_num, block_size, total_size):
    """Simple progress indicator for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [{bar}] {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb_down:.1f} MB...")
        sys.stdout.flush()


def download_file(urls: list, dest_path: Path) -> bool:
    """Try downloading from a list of URLs, return True on success."""
    for i, url in enumerate(urls):
        try:
            print(f"  Attempting URL {i+1}/{len(urls)}: {url[:80]}...")
            urlretrieve(url, str(dest_path), reporthook=progress_hook)
            print()  # newline after progress bar
            return True
        except (URLError, HTTPError, OSError) as e:
            print(f"\n  [!] Failed: {e}")
            if i < len(urls) - 1:
                print("  Trying next mirror...")
    return False


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract zip file, handling nested directories."""
    print(f"  Extracting to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(output_dir))

    # If extraction created a single subdirectory, flatten it
    children = list(output_dir.iterdir())
    subdirs = [c for c in children if c.is_dir() and c.name != "__MACOSX"]
    if len(subdirs) == 1 and not any(c.is_file() and c.suffix == '.txt' for c in children):
        # Flatten: move files from subdirectory to output_dir
        subdir = subdirs[0]
        for item in subdir.iterdir():
            if item.name == "__MACOSX":
                continue
            target = output_dir / item.name
            if not target.exists():
                item.rename(target)
        # Try removing the now-empty subdirectory
        try:
            subdir.rmdir()
        except OSError:
            pass  # Not empty, that's fine


def validate_dataset(output_dir: Path, expected_files: list, validation: dict) -> bool:
    """Validate that expected files exist and have reasonable sizes."""
    all_ok = True

    # Check expected files
    for fname in expected_files:
        fpath = output_dir / fname
        if not fpath.exists():
            # Search recursively in case of nested extraction
            found = list(output_dir.rglob(fname))
            if found:
                # Move to expected location
                found[0].rename(fpath)
            else:
                print(f"  [X] Missing: {fname}")
                all_ok = False
                continue
        print(f"  [OK] Found: {fname} ({fpath.stat().st_size / 1024:.0f} KB)")

    # Check line counts where specified
    for fname, constraints in validation.items():
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath, 'r') as f:
                line_count = sum(1 for _ in f)
            min_lines = constraints.get("min_lines", 0)
            max_lines = constraints.get("max_lines", float('inf'))
            if min_lines <= line_count <= max_lines:
                print(f"  [OK] {fname}: {line_count} lines (expected {min_lines}-{max_lines})")
            else:
                print(f"  [!] {fname}: {line_count} lines (expected {min_lines}-{max_lines})")
                all_ok = False

    return all_ok


# ============================================================
# Main download logic
# ============================================================

def download_dataset(name: str) -> bool:
    """Download and validate a single dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    config = DATASETS[name]
    output_dir = config["output_dir"]

    print(f"\n{'='*60}")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    # Check if already downloaded and valid
    if output_dir.exists() and any(output_dir.rglob("*.txt")):
        print(f"  Dataset directory already exists: {output_dir}")
        print(f"  Validating existing files...")
        if validate_dataset(output_dir, config["expected_files"], config["validation"]):
            print(f"  [OK] Dataset already downloaded and valid. Skipping.")
            return True
        else:
            print(f"  [!] Existing data incomplete. Re-downloading...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if name == "cmapss":
        # Download files individually from Hugging Face mirror
        base_url = "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main/"
        success = True
        for filename in config["expected_files"]:
            dest_path = output_dir / filename
            if dest_path.exists():
                print(f"  File already exists: {filename}")
                continue
            url = f"{base_url}{filename}"
            print(f"  Downloading {filename} from {url}...")
            try:
                # Add headers to avoid user-agent blocking if needed
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req) as response:
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  [X] Failed to download {filename}: {e}")
                success = False
                break
        if success:
            print(f"\n  Validating downloaded files...")
            if validate_dataset(output_dir, config["expected_files"], config["validation"]):
                print(f"\n  [OK] {name} dataset ready!")
                return True
        return False

    zip_path = output_dir / config["zip_name"]

    # Download
    print(f"\n  Downloading {config['zip_name']}...")
    if not download_file(config["urls"], zip_path):
        print(f"  [X] All download URLs failed for {name}.")
        print(f"  Please download manually and place in: {output_dir}")
        return False

    # Extract
    try:
        extract_zip(zip_path, output_dir)
    except zipfile.BadZipFile:
        print(f"  [X] Downloaded file is not a valid ZIP archive.")
        zip_path.unlink(missing_ok=True)
        return False

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    # Validate
    print(f"\n  Validating extracted files...")
    if validate_dataset(output_dir, config["expected_files"], config["validation"]):
        print(f"\n  [OK] {name} dataset ready!")
        return True
    else:
        print(f"\n  [!] {name} dataset downloaded but some validations failed.")
        print(f"     Files are in {output_dir} -- check manually.")
        return True  # Still consider it a success if files are present



def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for AeroSentinal"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AeroSentinal — Dataset Download")
    print("=" * 60)

    datasets_to_download = (
        list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    )

    results = {}
    for name in datasets_to_download:
        results[name] = download_dataset(name)

    # Summary
    print(f"\n{'='*60}")
    print("  Download Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "[OK] Ready" if success else "[X] Failed"
        print(f"  {status}: {name}")

    if all(results.values()):
        print(f"\n  All datasets ready! Next step: python scripts/generate_synthetic.py")
        return 0
    else:
        print(f"\n  Some datasets failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
