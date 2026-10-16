"""
Step 9 — Model comparison CLI.
Aggregates PatchCore, EfficientAD, and PaDiM metrics into CSV / JSON / Markdown reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import comparison submodule directly to avoid pulling optional heavy deps via package __init__
from src.evaluation.comparison import (  # noqa: E402
    collect_benchmark_rows,
    render_markdown_table,
    write_csv_report,
    write_json_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CompareModels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anomaly detection model benchmarks")
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "metrics"),
        help="Directory containing <model>_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports"),
        help="Directory for comparison artifacts",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["patchcore", "efficientad", "padim"],
        choices=["patchcore", "efficientad", "padim"],
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    rows = collect_benchmark_rows(metrics_dir, models=args.models)

    missing = [r.model for r in rows if r.missing]
    if missing:
        logger.warning(
            "Missing evaluation reports for: %s. Run evaluate.py for each model first.",
            ", ".join(missing),
        )

    context = {
        "dataset": "AutoVI (automotive visual inspection)",
        "category": "bracket_assembly",
        "note": "Values sourced only from existing metrics JSON; missing models are not estimated.",
    }

    md_path = output_dir / "model_comparison.md"
    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"

    md_text = render_markdown_table(rows, title="Quality Inspector — Model Comparison (Step 9)")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    write_csv_report(rows, csv_path)
    write_json_report(rows, json_path, extra=context)

    print("\n" + "=" * 78)
    print("  MODEL COMPARISON TABLE (from recorded experiments only)")
    print("=" * 78)
    print(f"{'Model':<14} | {'AUROC':<8} | {'F1':<8} | {'Pix-AUROC':<10} | {'Latency':<10} | {'FPS':<6} | {'VRAM':<8}")
    print("-" * 78)
    for row in rows:
        if row.missing:
            na = "N/A"
            print(f"{row.model:<14} | {na:<8} | {na:<8} | {na:<10} | {na:<10} | {na:<6} | {na:<8}")
            continue
        print(
            f"{row.model:<14} | "
            f"{row.image_auroc:<8.4f} | "
            f"{row.f1_score:<8.4f} | "
            f"{row.pixel_auroc:<10.4f} | "
            f"{row.mean_latency_ms:<7.1f} ms | "
            f"{row.fps:<6.1f} | "
            f"{row.vram_usage_mb:<8.1f}"
        )
    print("=" * 78)
    print(f"Markdown: {md_path}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    if missing:
        print("\n[INFO] Complete missing runs, e.g.:")
        for m in missing:
            print(f"  python scripts/evaluate.py --model {m}")
    else:
        print("\n[SUCCESS] All requested models have evaluation reports.")


if __name__ == "__main__":
    main()
