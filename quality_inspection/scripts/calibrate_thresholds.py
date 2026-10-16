"""
Threshold Calibration Script for Quality Decision Engine.
Uses validation/test data to determine optimal PASS/REVIEW/FAIL thresholds.
Avoids arbitrary threshold selection by using data-driven calibration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AutomotiveAnomalyDataset
from src.inference.inferencer import PatchCoreInferencer
from src.postprocessing.decision_engine import DecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CalibrateThresholds")


def main():
    parser = argparse.ArgumentParser(description="Calibrate PASS/REVIEW/FAIL thresholds")
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=["patchcore", "efficientad", "padim"],
        help="Model to use for calibration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/<model>/checkpoint)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly"),
        help="Path to dataset for calibration",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="adaptive",
        choices=["f1_optimal", "precision_target", "percentile", "adaptive"],
        help="Calibration method",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports" / "decision_calibration.json"),
        help="Path to save calibration report",
    )
    args = parser.parse_args()

    # Load model
    default_ckpts = {
        "patchcore": PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt",
        "efficientad": PROJECT_ROOT / "models" / "efficientad" / "efficientad_checkpoint.pt",
        "padim": PROJECT_ROOT / "models" / "padim" / "padim_checkpoint.pt",
    }
    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_ckpts[args.model]

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading model: {args.model} from {ckpt_path}")
    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)

    # Load test data for calibration
    logger.info(f"Loading calibration data from: {args.data_dir}")
    test_dataset = AutomotiveAnomalyDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=(256, 256),
        is_training=False,
    )

    logger.info(f"Calibration dataset: {len(test_dataset)} samples")

    # Extract anomaly scores
    logger.info("Extracting anomaly scores from calibration dataset...")
    anomaly_scores = []
    labels = []

    for sample in test_dataset.samples:
        result = inferencer.predict(sample["image_path"])
        anomaly_scores.append(result["anomaly_score"])
        labels.append(sample["label"])

    logger.info(f"Extracted {len(anomaly_scores)} scores (nominal: {sum(1 for l in labels if l == 0)}, anomalous: {sum(1 for l in labels if l == 1)})")

    # Calibrate thresholds
    logger.info(f"Calibrating thresholds using method: {args.method}")
    thresholds, report = DecisionEngine.calibrate_from_validation(
        anomaly_scores=anomaly_scores,
        labels=labels,
        method=args.method,
        metadata={
            "model": args.model,
            "checkpoint": str(ckpt_path),
            "dataset": "autovi",
            "category": "bracket_assembly",
            "n_samples": len(anomaly_scores),
        },
    )

    # Create decision engine with calibrated thresholds
    engine = DecisionEngine(thresholds=thresholds, calibration_report=report)

    # Save calibration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_calibration(output_path)

    # Evaluate calibrated decisions
    stats = engine.get_decision_stats(anomaly_scores, labels)

    # Print summary
    print("\n" + "=" * 80)
    print("  THRESHOLD CALIBRATION REPORT")
    print("=" * 80)
    print(f"Model:                    {args.model}")
    print(f"Calibration Method:       {report.method}")
    print(f"Validation AUROC:         {report.validation_auc:.4f}")
    print("-" * 80)
    print("DECISION THRESHOLDS:")
    print(f"  PASS threshold:         < {report.pass_threshold:.4f}")
    print(f"  REVIEW range:           [{report.pass_threshold:.4f}, {report.fail_threshold:.4f})")
    print(f"  FAIL threshold:         >= {report.fail_threshold:.4f}")
    print("-" * 80)
    print("CALIBRATION METRICS:")
    print(f"  PASS precision:          {report.pass_precision:.4f}")
    print(f"  PASS recall:            {report.pass_recall:.4f}")
    print(f"  FAIL precision:         {report.fail_precision:.4f}")
    print(f"  FAIL recall:            {report.fail_recall:.4f}")
    print("-" * 80)
    print("DECISION DISTRIBUTION (on calibration set):")
    for decision, count in stats["decision_counts"].items():
        pct = count / stats["total_samples"] * 100
        print(f"  {decision:8s}: {count:3d} ({pct:5.1f}%)")
    print("=" * 80)
    print(f"Calibration saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
