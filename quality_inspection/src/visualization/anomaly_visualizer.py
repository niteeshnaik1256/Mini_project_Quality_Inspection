"""
Industrial Anomaly Visualization & Evidence Generation Engine.
Constructs the complete 6-panel inspection panel:
[Input Image] -> [Anomaly Map] -> [Ground Truth] -> [Binary Mask] -> [Overlay] -> [Bounding Box]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.postprocessing.localizer import DefectBoundingBox, LocalizationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnomalyVisualizer")


class AnomalyVisualizer:
    """Generates industrial optical inspection evidence panels and overlays."""

    @staticmethod
    def draw_defect_bounding_boxes(
        image_rgb: np.ndarray,
        boxes: List[DefectBoundingBox],
        color: Tuple[int, int, int] = (255, 0, 0),  # Bright Red
        thickness: int = 2,
    ) -> np.ndarray:
        """Draws crisp bounding boxes, defect labels, and centroid markers on RGB image."""
        annotated = image_rgb.copy()
        for idx, box in enumerate(boxes):
            # Box rectangle
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, thickness)

            # Centroid crosshair
            cx, cy = box.centroid
            cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=2)

            # Text label badge
            label = f"DEFECT #{idx+1} ({box.peak_score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Label background box
            cv2.rectangle(
                annotated,
                (box.x1, max(0, box.y1 - th - 6)),
                (box.x1 + tw + 4, max(th + 6, box.y1)),
                (40, 40, 40),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (box.x1 + 2, max(th + 2, box.y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def generate_inspection_evidence_panel(
        self,
        image_input: Union[str, Path, np.ndarray],
        anomaly_map: np.ndarray,
        localization_result: LocalizationResult,
        ground_truth_mask: Optional[np.ndarray] = None,
        title_prefix: str = "INSPECTION EVIDENCE",
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Renders the full industrial 6-stage evidence panel:
        [1. Original Image]
        [2. Anomaly Heatmap (Continuous)]
        [3. Ground-Truth Mask]
        [4. Predicted Binary Mask (Thresholded)]
        [5. Heatmap Alpha Overlay]
        [6. Defect Bounding Box Annotation]
        """
        # Load / normalize RGB image
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            sample_name = Path(image_input).name
        else:
            rgb = image_input.copy()
            sample_name = "inspection_frame"

        h, w = rgb.shape[:2]

        # 1. Normalized Colormap for Heatmap
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        norm_uint8 = (norm_map * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 2. Alpha Overlay
        overlay = cv2.addWeighted(rgb, 0.65, heatmap_colored_rgb, 0.35, 0)

        # 3. Ground Truth Mask handling
        if ground_truth_mask is not None:
            gt_disp = (ground_truth_mask > 0).astype(np.uint8) * 255
        else:
            gt_disp = np.zeros((h, w), dtype=np.uint8)

        # 4. Predicted Binary Mask
        pred_bin = localization_result.binary_mask

        # 5. Bounding Box Annotated Image
        annotated_rgb = self.draw_defect_bounding_boxes(
            rgb,
            localization_result.bounding_boxes,
        )

        # Build 6-panel Figure
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.8), dpi=130)

        panels = [
            (rgb, "1. Input Component", None),
            (anomaly_map, f"2. Anomaly Map\n(Peak: {localization_result.peak_anomaly_score:.2f})", "jet"),
            (gt_disp, "3. Ground-Truth Mask", "gray"),
            (pred_bin, f"4. Predicted Defect Mask\n(Threshold: {localization_result.threshold_applied:.2f})", "gray"),
            (overlay, "5. Heatmap Overlay", None),
            (annotated_rgb, f"6. Defect BBox ({localization_result.num_defects} found)", None),
        ]

        decision_tag = "FAIL (DEFECT)" if localization_result.is_anomalous else "PASS (NOMINAL)"
        tag_color = "#d95f02" if localization_result.is_anomalous else "#2ca02c"

        for idx, (img_data, p_title, cmap) in enumerate(panels):
            if cmap is not None:
                im = axes[idx].imshow(img_data, cmap=cmap)
                if idx == 1:
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            else:
                axes[idx].imshow(img_data)

            axes[idx].set_title(p_title, fontsize=10, fontweight="bold", pad=8)
            axes[idx].axis("off")

        fig.suptitle(
            f"{title_prefix} | Part: {sample_name} | Decision: {decision_tag} | Max Score: {localization_result.peak_anomaly_score:.3f}",
            fontsize=13,
            fontweight="bold",
            color=tag_color,
            y=1.03,
        )

        plt.tight_layout()

        if output_path is None:
            output_path = Path("results/visualizations/evidence") / f"{Path(sample_name).stem}_evidence.png"

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved inspection evidence panel to: {out}")
        return out
