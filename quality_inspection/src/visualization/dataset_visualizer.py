"""
Industrial Dataset Visualization Module.
Renders comparative galleries of normal components, anomalous parts,
ground-truth defect masks, and defect distribution charts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless rendering for server/CI/headless execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DatasetVisualizer")


class DatasetVisualizer:
    """Generates visual audit figures and defect localization previews."""

    def __init__(self, category_dir: str | Path):
        self.category_dir = Path(category_dir).resolve()
        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category folder not found: {self.category_dir}")

    def render_sample_inspection_grid(
        self,
        output_path: str | Path,
        max_defect_types: int = 4,
    ) -> Path:
        """
        Creates an industrial inspection gallery showing:
        Row per defect type:
          [Nominal Train] | [Nominal Test] | [Defect Test Image] | [Ground Truth Mask] | [Defect Overlay]
        """
        train_good = sorted(list((self.category_dir / "train" / "good").glob("*.png")))
        test_good = sorted(list((self.category_dir / "test" / "good").glob("*.png")))
        test_dir = self.category_dir / "test"
        gt_dir = self.category_dir / "ground_truth"

        defect_types = [
            d.name for d in sorted(test_dir.iterdir()) if d.is_dir() and d.name != "good"
        ][:max_defect_types]

        if not defect_types:
            raise ValueError(f"No defect folders found in {test_dir}")

        num_rows = len(defect_types)
        fig, axes = plt.subplots(num_rows, 5, figsize=(16, 3.2 * num_rows), dpi=120)
        if num_rows == 1:
            axes = np.expand_dims(axes, 0)

        col_titles = [
            "Reference Part (Train Good)",
            "Unseen Part (Test Good)",
            "Defective Inspection Image",
            "Ground-Truth Mask",
            "Defect Localization Overlay",
        ]
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=10)

        for row_idx, d_name in enumerate(defect_types):
            # 1. Train Good
            t_good_img = Image.open(train_good[row_idx % len(train_good)]).convert("RGB")
            axes[row_idx, 0].imshow(t_good_img)
            axes[row_idx, 0].set_ylabel(f"Archetype:\n{d_name}", fontsize=10, fontweight="bold")
            axes[row_idx, 0].axis("off")

            # 2. Test Good
            u_good_img = Image.open(test_good[row_idx % len(test_good)]).convert("RGB")
            axes[row_idx, 1].imshow(u_good_img)
            axes[row_idx, 1].axis("off")

            # 3. Defect Image
            defect_imgs = sorted(list((test_dir / d_name).glob("*.png")))
            d_img_path = defect_imgs[0]
            d_img = Image.open(d_img_path).convert("RGB")
            axes[row_idx, 2].imshow(d_img)
            axes[row_idx, 2].axis("off")

            # 4. Ground Truth Mask
            mask_path = gt_dir / d_name / f"{d_img_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = gt_dir / d_name / d_img_path.name

            if mask_path.exists():
                gt_mask = Image.open(mask_path).convert("L")
                axes[row_idx, 3].imshow(gt_mask, cmap="gray")
            else:
                gt_mask = Image.new("L", d_img.size, 0)
                axes[row_idx, 3].imshow(gt_mask, cmap="gray")
            axes[row_idx, 3].axis("off")

            # 5. Red Overlay
            d_arr = np.array(d_img).astype(np.float32)
            m_arr = np.array(gt_mask).astype(np.float32) / 255.0
            overlay = d_arr.copy()
            # Highlight defective pixels with bright red tint
            overlay[..., 0] = np.clip(overlay[..., 0] + m_arr * 180, 0, 255)
            overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - m_arr * 0.6), 0, 255)
            overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - m_arr * 0.6), 0, 255)
            axes[row_idx, 5 - 1].imshow(overlay.astype(np.uint8))
            axes[row_idx, 5 - 1].axis("off")

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Sample inspection grid saved to: {out}")
        return out

    def render_defect_distribution(
        self,
        distribution: Dict[str, int],
        output_path: str | Path,
    ) -> Path:
        """Plots defect class counts and imbalance proportions."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        labels = list(distribution.keys())
        counts = list(distribution.values())
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"][: len(labels)]

        bars = ax.bar(labels, counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
        ax.set_title("Automotive Defect Archetype Distribution (Test Partition)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Defect Sample Count", fontsize=11)
        ax.set_xlabel("Defect Type", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Defect distribution chart saved to: {out}")
        return out
