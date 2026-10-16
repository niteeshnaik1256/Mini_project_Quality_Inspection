"""
Industrial Anomaly Localization & Evidence Generation CLI Script.
Processes test components, applies calibrated morphological thresholding,
extracts defect bounding boxes, and generates 6-panel audit evidence panels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AutomotiveAnomalyDataset
from src.inference.inferencer import PatchCoreInferencer
from src.postprocessing.localizer import DefectLocalizer
from src.visualization.anomaly_visualizer import AnomalyVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VisualizeAnomaliesCLI")


def main():
    parser = argparse.ArgumentParser(description="Generate 6-Panel Industrial Anomaly Evidence Panels")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly"),
        help="Path to category directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly threshold (if None, reads optimal threshold from metrics JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations" / "evidence"),
        help="Directory to save evidence panels",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Run training first.")
        sys.exit(1)

    # 1. Determine Threshold
    threshold = args.threshold
    if threshold is None:
        metrics_json = PROJECT_ROOT / "results" / "metrics" / "patchcore_metrics.json"
        if metrics_json.exists():
            with open(metrics_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            threshold = float(data.get("optimal_threshold", 2.0871))
            logger.info(f"Loaded calibrated threshold from metrics report: {threshold:.4f}")
        else:
            threshold = 2.0871
            logger.info(f"Using default threshold: {threshold:.4f}")

    # 2. Initialize Inferencer, Localizer, Visualizer
    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)
    localizer = DefectLocalizer(threshold=threshold, min_contour_area=15)
    visualizer = AnomalyVisualizer()

    # 3. Load Test Dataset
    test_ds = AutomotiveAnomalyDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=(256, 256),
        is_training=False,
    )

    print("\n" + "=" * 90)
    print("  INDUSTRIAL DEFECT LOCALIZATION & EVIDENCE GENERATION AUDIT")
    print(f"  Operating Decision Threshold: {threshold:.4f}")
    print("=" * 90)
    print(f"{'Archetype':<16} | {'Sample':<8} | {'Score':<7} | {'Defects':<7} | {'Primary BBox [x1,y1,x2,y2]':<26} | {'Area (px)':<9}")
    print("-" * 90)

    # Pick 1 sample per distinct defect folder + 1 nominal
    processed_archetypes = set()
    evidence_paths = []

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        d_type = sample["defect_type"]
        if d_type in processed_archetypes:
            continue
        processed_archetypes.add(d_type)

        img_path = sample["image_path"]
        gt_mask = sample["mask"].squeeze(0).numpy()

        # Run Inference
        inf_res = inferencer.predict(img_path)
        anomaly_map = inf_res["anomaly_map"]

        # Run Localization
        loc_res = localizer.localize(anomaly_map, threshold=threshold)

        # Render 6-Panel Evidence Panel
        panel_filename = f"{d_type}_{Path(img_path).stem}_evidence.png"
        panel_path = out_dir / panel_filename
        visualizer.generate_inspection_evidence_panel(
            image_input=img_path,
            anomaly_map=anomaly_map,
            localization_result=loc_res,
            ground_truth_mask=gt_mask,
            title_prefix="AOI EVIDENCE AUDIT",
            output_path=panel_path,
        )
        evidence_paths.append(panel_path)

        # Format primary bounding box
        if loc_res.bounding_boxes:
            primary_box = loc_res.bounding_boxes[0]
            bbox_str = f"[{primary_box.x1}, {primary_box.y1}, {primary_box.x2}, {primary_box.y2}]"
            area_str = f"{primary_box.area_pixels:.1f}"
        else:
            bbox_str = "None (Defect-Free)"
            area_str = "0.0"

        print(f"{d_type:<16} | {Path(img_path).name:<8} | {loc_res.peak_anomaly_score:<7.3f} | {loc_res.num_defects:<7d} | {bbox_str:<26} | {area_str:<9}")

    print("=" * 90)
    print(f"\n[SUCCESS] Generated {len(evidence_paths)} industrial evidence panels in:")
    print(f"  --> {out_dir}")


if __name__ == "__main__":
    main()
