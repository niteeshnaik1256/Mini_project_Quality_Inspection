"""
PatchCore Inference CLI Tool.
Performs single-image and batch industrial quality inspections,
computes inference latency, and saves 2D localization heatmaps.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inferencer import PatchCoreInferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InferCLI")


def save_heatmap_visualization(
    original_img_path: Path,
    anomaly_map: np.ndarray,
    anomaly_score: float,
    output_path: Path,
) -> None:
    """Save original image and anomaly heatmap side-by-side."""
    orig_bgr = cv2.imread(str(original_img_path))
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Normalize heatmap to [0, 1] for visual display
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    norm_map_uint8 = (norm_map * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Blend overlay
    overlay = cv2.addWeighted(orig_rgb, 0.65, color_map_rgb, 0.35, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title(f"Anomaly Heatmap (Score: {anomaly_score:.3f})", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run Industrial PatchCore Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt"),
        help="Path to trained PatchCore checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for inspection",
    )
    parser.add_argument(
        "--test-split",
        action="store_true",
        default=True,
        help="Run inference across representative test samples (nominal and defects)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "heatmaps"),
        help="Directory to save generated heatmaps",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Run training first.")
        sys.exit(1)

    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        # Single image evaluation
        img_path = Path(args.image)
        logger.info(f"Running inspection on: {img_path}")
        result = inferencer.predict(img_path)

        out_heatmap = output_dir / f"{img_path.stem}_heatmap.png"
        save_heatmap_visualization(img_path, result["anomaly_map"], result["anomaly_score"], out_heatmap)

        print("\n" + "=" * 55)
        print("  SINGLE PART INSPECTION RESULT")
        print("=" * 55)
        print(f"Image:         {img_path.name}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Latency:       {result['latency_ms']:.1f} ms  ({result['fps']:.1f} FPS)")
        print(f"Heatmap:       {out_heatmap}")
        print("=" * 55)

    elif args.test_split:
        # Evaluate representative samples from each defect partition
        test_root = PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly" / "test"
        if not test_root.exists():
            logger.error(f"Test directory not found: {test_root}")
            sys.exit(1)

        target_folders = sorted([d for d in test_root.iterdir() if d.is_dir()])
        print("\n" + "=" * 75)
        print("  PATCHCORE INFERENCE BENCHMARK ACROSS AUTOMOTIVE TEST PARTITIONS")
        print("=" * 75)
        print(f"{'Partition / Archetype':<24} | {'Sample':<10} | {'Score':<8} | {'Latency':<9} | {'FPS':<6}")
        print("-" * 75)

        for folder in target_folders:
            imgs = sorted(list(folder.glob("*.png")))
            if not imgs:
                continue
            sample_img = imgs[0]  # Take first sample
            res = inferencer.predict(sample_img)

            out_heatmap = output_dir / f"{folder.name}_{sample_img.stem}_heatmap.png"
            save_heatmap_visualization(sample_img, res["anomaly_map"], res["anomaly_score"], out_heatmap)

            label_tag = f"[{folder.name}]"
            print(f"{label_tag:<24} | {sample_img.name:<10} | {res['anomaly_score']:<8.4f} | {res['latency_ms']:<6.1f} ms | {res['fps']:<6.1f}")

        print("=" * 75)
        print(f"Localized anomaly heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
