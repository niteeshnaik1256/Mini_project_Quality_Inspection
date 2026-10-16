"""
Command-line Dataset Audit & Inspection Script for AutoVI.
Executes structural validation, mask pairing checks, train/test leakage scans,
defect distribution profiling, and exports summary JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspector import DatasetInspector


def main():
    parser = argparse.ArgumentParser(description="Inspect and Audit Industrial Automotive Anomaly Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi"),
        help="Path to AutoVI raw dataset directory",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bracket_assembly",
        help="Category subfolder to inspect",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports" / "dataset_summary.json"),
        help="Path to save JSON audit report",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    cat_dir = data_dir / args.category if (data_dir / args.category).exists() else data_dir

    if not cat_dir.exists():
        print(f"[ERROR] Target dataset directory does not exist: {cat_dir}")
        print("Please run `python quality_inspection/scripts/prepare_data.py` first.")
        sys.exit(1)

    inspector = DatasetInspector(cat_dir)
    summary = inspector.inspect_category()
    inspector.print_report(summary)
    inspector.save_summary(summary, args.output_json)

    if summary.train_test_leakage_detected:
        print("\n[CRITICAL ERROR] Train/Test data leakage detected! Cannot proceed with training until resolved.")
        sys.exit(2)
    elif not summary.is_valid_anomalib_structure:
        print("\n[WARNING] Dataset structure has validation errors. Review report above.")
        sys.exit(3)
    else:
        print("\n[VERIFIED] Dataset structure conforms to one-class anomaly detection standard.")


if __name__ == "__main__":
    main()
