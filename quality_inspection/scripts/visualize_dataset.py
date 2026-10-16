"""
Dataset Visualization CLI script.
Generates sample inspection panels and defect distribution plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.dataset_visualizer import DatasetVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize Automotive Inspection Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly"),
        help="Path to category directory",
    )
    parser.add_argument(
        "--output-grid",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations" / "dataset_samples.png"),
        help="Output image path for sample grid",
    )
    parser.add_argument(
        "--output-dist",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations" / "defect_distribution.png"),
        help="Output image path for defect distribution bar chart",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports" / "dataset_summary.json"),
        help="Path to existing dataset summary JSON",
    )

    args = parser.parse_args()
    cat_dir = Path(args.data_dir)
    if not cat_dir.exists():
        print(f"[ERROR] Category directory not found: {cat_dir}")
        sys.exit(1)

    visualizer = DatasetVisualizer(cat_dir)

    # 1. Render sample inspection grid
    grid_path = visualizer.render_sample_inspection_grid(args.output_grid)
    print(f"[SUCCESS] Sample inspection gallery created: {grid_path}")

    # 2. Render defect distribution chart
    summary_file = Path(args.summary_json)
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        dist = summary_data.get("defect_distribution", {})
    else:
        # Fallback count
        test_dir = cat_dir / "test"
        dist = {
            d.name: len(list(d.glob("*.png")))
            for d in test_dir.iterdir()
            if d.is_dir() and d.name != "good"
        }

    dist_path = visualizer.render_defect_distribution(dist, args.output_dist)
    print(f"[SUCCESS] Defect distribution plot created: {dist_path}")


if __name__ == "__main__":
    main()
