"""
Manual ROI extraction for automotive component inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.roi.config import ROIRectangle


@dataclass
class ROICropResult:
    """Cropped ROI and mapping metadata back to full-frame coordinates."""

    crop_bgr: np.ndarray
    crop_box_xyxy: Tuple[int, int, int, int]
    offset_xy: Tuple[int, int]


class ManualROICropper:
    """Apply configured axis-aligned ROI to BGR images."""

    def __init__(self, rectangle: ROIRectangle):
        self.rectangle = rectangle

    def crop(self, image_bgr: np.ndarray) -> ROICropResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {image_bgr.shape}")

        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = self.rectangle.resolve(w, h)
        crop = image_bgr[y1:y2, x1:x2].copy()
        return ROICropResult(
            crop_bgr=crop,
            crop_box_xyxy=(x1, y1, x2, y2),
            offset_xy=(x1, y1),
        )

    @staticmethod
    def paste_heatmap_to_full_frame(
        full_shape: Tuple[int, int],
        roi_heatmap: np.ndarray,
        crop_box_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Embed ROI-sized heatmap into a full-resolution canvas (zeros outside ROI).
        """
        full_h, full_w = full_shape
        x1, y1, x2, y2 = crop_box_xyxy
        roi_h, roi_w = y2 - y1, x2 - x1
        if roi_heatmap.shape[:2] != (roi_h, roi_w):
            roi_heatmap = cv2.resize(roi_heatmap, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((full_h, full_w), dtype=np.float32)
        canvas[y1:y2, x1:x2] = roi_heatmap
        return canvas
