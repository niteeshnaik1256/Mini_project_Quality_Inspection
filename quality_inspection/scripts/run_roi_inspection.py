"""
Step 10 — ROI-constrained quality inspection CLI.
Camera/Image → ROI crop → anomaly model → heatmap (full frame + ROI overlay).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inferencer import PatchCoreInferencer
from src.roi.config import load_roi_config
from src.roi.pipeline import ROIInspectionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ROIInspectionCLI")


def save_roi_visualization(
    image_bgr: np.ndarray,
    result,
    output_path: Path,
) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = result.roi_box_xyxy
    vis = rgb.copy()
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

    heat = result.anomaly_map_full
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.6, color_rgb, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    axes[0].imshow(vis)
    axes[0].set_title("ROI Box")
    axes[0].axis("off")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title(f"Full-frame Heatmap ({result.anomaly_score:.3f})")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROI-constrained anomaly inspection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "roi.yaml"),
    )
    parser.add_argument("--image", type=str, required=True, help="Input automotive image path")
    parser.add_argument("--product-id", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations" / "roi_inspection.png"),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    roi_cfg = load_roi_config(cfg_path)
    ckpt_rel = raw_cfg.get("inspection", {}).get("checkpoint", "models/patchcore/patchcore_checkpoint.pt")
    ckpt_path = Path(ckpt_rel)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)
    pipeline = ROIInspectionPipeline(inferencer=inferencer, roi_config=roi_cfg)

    img_path = Path(args.image)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        logger.error("Could not read image: %s", img_path)
        sys.exit(1)

    result = pipeline.inspect(bgr, product_id=args.product_id)
    out_path = Path(args.output)
    save_roi_visualization(bgr, result, out_path)

    print("\n" + "=" * 60)
    print("  ROI QUALITY INSPECTION")
    print("=" * 60)
    print(f"Image:          {img_path.name}")
    print(f"Product ID:     {result.product_id}")
    print(f"ROI (xyxy):     {result.roi_box_xyxy}")
    print(f"Anomaly score:  {result.anomaly_score:.4f}")
    print(f"Latency:        {result.latency_ms:.1f} ms ({result.fps:.1f} FPS)")
    print(f"Visualization:  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
