"""
Comprehensive Model Evaluation CLI Tool.
Executes test partition evaluation, calculates Image-level and Pixel-level metrics,
benchmarks inference latency, plots ROC curve, and outputs structured audit JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AutomotiveAnomalyDataset
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.experiment_log import append_experiment_record
from src.inference.inferencer import PatchCoreInferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvaluateCLI")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=["patchcore", "efficientad", "padim"],
        help="Target model architecture to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (defaults to models/<model>/ checkpoint)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly"),
        help="Path to category directory containing test/ and ground_truth/",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON report",
    )
    parser.add_argument(
        "--output-roc",
        type=str,
        default=None,
        help="Path to save ROC curve plot",
    )

    args = parser.parse_args()
    model_name = args.model.lower()
    default_ckpts = {
        "patchcore": PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt",
        "efficientad": PROJECT_ROOT / "models" / "efficientad" / "efficientad_checkpoint.pt",
        "padim": PROJECT_ROOT / "models" / "padim" / "padim_checkpoint.pt",
    }
    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_ckpts[model_name]

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Please train the model first.")
        sys.exit(1)

    out_metrics = (
        Path(args.output_metrics)
        if args.output_metrics
        else PROJECT_ROOT / "results" / "metrics" / f"{model_name}_metrics.json"
    )
    out_roc = (
        Path(args.output_roc)
        if args.output_roc
        else PROJECT_ROOT / "results" / "visualizations" / f"{model_name}_roc_curve.png"
    )

    # 1. Initialize Test Dataset
    test_ds = AutomotiveAnomalyDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=(256, 256),
        is_training=False,
    )

    # 2. Initialize Inferencer
    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)

    # 3. Run Benchmark Evaluation
    evaluator = ModelEvaluator(
        inferencer=inferencer,
        test_dataset=test_ds,
        model_name=model_name,
    )
    report = evaluator.evaluate()

    # 4. Save JSON Report
    evaluator.save_report(report, out_metrics)

    experiment_log = PROJECT_ROOT / "results" / "metrics" / "experiment_log.jsonl"
    append_experiment_record(
        experiment_log,
        report,
        dataset_version="autovi",
        category="bracket_assembly",
        checkpoint_path=str(ckpt_path),
    )
    logger.info("Appended experiment record to %s", experiment_log)

    # 5. Extract scores and plot ROC curve
    y_true = [s["label"] for s in test_ds.samples]
    # Re-extract cached predictions from report or run evaluation
    y_scores = [inferencer.predict(s["image_path"])["anomaly_score"] for s in test_ds.samples]
    import numpy as np
    evaluator.plot_roc_curve(np.array(y_true), np.array(y_scores), out_roc)

    # 6. Render Industrial Summary Table
    cls_m = report["classification_metrics"]
    prod_m = report["production_metrics"]

    print("\n" + "=" * 75)
    print(f"  INDUSTRIAL AOI BENCHMARK AUDIT: [{model_name.upper()}]")
    print("=" * 75)
    print(f"Total Test Components:       {report['samples_evaluated']} (Nominal: {report['nominal_samples']}, Defective: {report['anomalous_samples']})")
    print("-" * 75)
    print("ACCURACY & LOCALIZATION PERFORMANCE:")
    print(f"  * Image-Level AUROC:       {report['image_auroc']:.4f}  ({report['image_auroc']*100:.2f}%)")
    print(f"  * Pixel-Level AUROC:       {report['pixel_auroc']:.4f}  ({report['pixel_auroc']*100:.2f}%)")
    print(f"  * Optimal Anomaly Threshold:{report['optimal_threshold']:.4f}")
    print(f"  * F1-Score:                {cls_m['f1_score']:.4f}")
    print(f"  * Precision:               {cls_m['precision']:.4f}  (TP: {cls_m['true_positives']}, FP: {cls_m['false_positives']})")
    print(f"  * Recall (Detection Rate): {cls_m['recall']:.4f}  (FN: {cls_m['false_negatives']})")
    print(f"  * Specificity:             {cls_m['specificity']:.4f}  (TN: {cls_m['true_negatives']})")
    print(f"  * False Positive Rate:     {cls_m['false_positive_rate']:.4f}")
    print(f"  * False Negative Rate:     {cls_m['false_negative_rate']:.4f}")
    print("-" * 75)
    print("PRODUCTION LATENCY & SYSTEM RESOURCES:")
    print(f"  * Mean Latency per Frame:  {prod_m['mean_latency_ms']:.1f} ms")
    print(f"  * 95th Percentile Latency: {prod_m['p95_latency_ms']:.1f} ms")
    print(f"  * Throughput (FPS):        {prod_m['fps']:.1f} FPS")
    print(f"  * Process RAM Usage:       {prod_m['ram_usage_mb']:.1f} MB")
    print(f"  * Peak VRAM Usage:         {prod_m['vram_usage_mb']:.1f} MB")
    print(f"  * Compute Device:          {prod_m['device'].upper()}")
    print("-" * 75)
    print(f"Metrics Report:  {out_metrics}")
    print(f"ROC Curve Plot:  {out_roc}")
    print("=" * 75)


if __name__ == "__main__":
    main()
