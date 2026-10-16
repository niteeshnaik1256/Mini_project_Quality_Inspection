"""
Industrial Defect Localization & Bounding Box Extraction Engine.
Converts continuous anomaly heatmaps into binary defect masks and extracts
bounding boxes, centroids, and defect area statistics via connected component analysis.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DefectLocalizer")


@dataclass
class DefectBoundingBox:
    """Bounding box and geometric properties of an isolated industrial defect."""
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area_pixels: float
    peak_score: float
    centroid: Tuple[int, int]

    def to_xyxy(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> List[int]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class LocalizationResult:
    """Complete spatial localization report for an inspection image."""
    binary_mask: np.ndarray
    bounding_boxes: List[DefectBoundingBox]
    num_defects: int
    peak_anomaly_score: float
    is_anomalous: bool
    threshold_applied: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_defects": self.num_defects,
            "peak_anomaly_score": round(self.peak_anomaly_score, 4),
            "is_anomalous": self.is_anomalous,
            "threshold_applied": round(self.threshold_applied, 4),
            "bounding_boxes": [asdict(b) for b in self.bounding_boxes],
        }


class DefectLocalizer:
    """
    Transforms continuous anomaly maps into actionable geometric bounding boxes.
    Applies morphological filtering and connected component contour segmentation.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        min_contour_area: int = 15,
        morph_kernel_size: int = 3,
    ):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.morph_kernel_size = morph_kernel_size
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )

    def localize(
        self,
        anomaly_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> LocalizationResult:
        """
        Thresholds anomaly map and extracts bounding boxes around defect regions.
        Args:
            anomaly_map: 2D float array (H, W)
            threshold: Optional override for anomaly decision threshold
        Returns:
            LocalizationResult containing binary mask and bounding boxes
        """
        active_thresh = threshold if threshold is not None else self.threshold
        h, w = anomaly_map.shape

        # 1. Binarize continuous anomaly map
        raw_binary = (anomaly_map >= active_thresh).astype(np.uint8) * 255

        # 2. Morphological cleanup (Open to remove speckle, Close to bridge small gaps)
        opened = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, self.kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)

        # 3. Contour Extraction
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes: List[DefectBoundingBox] = []
        clean_mask = np.zeros_like(closed)

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_contour_area:
                continue

            # Draw valid contour into clean binary mask
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

            # Bounding box
            x, y, bw, bh = cv2.boundingRect(cnt)
            roi_scores = anomaly_map[y : y + bh, x : x + bw]
            peak_val = float(np.max(roi_scores)) if roi_scores.size > 0 else float(anomaly_map[y, x])

            # Centroid
            moments = cv2.moments(cnt)
            cx = int(moments["m10"] / (moments["m00"] + 1e-8))
            cy = int(moments["m01"] / (moments["m00"] + 1e-8))

            bounding_boxes.append(
                DefectBoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh,
                    width=bw,
                    height=bh,
                    area_pixels=area,
                    peak_score=peak_val,
                    centroid=(cx, cy),
                )
            )

        peak_score = float(np.max(anomaly_map))
        is_anomalous = len(bounding_boxes) > 0 or peak_score >= active_thresh

        return LocalizationResult(
            binary_mask=clean_mask,
            bounding_boxes=bounding_boxes,
            num_defects=len(bounding_boxes),
            peak_anomaly_score=peak_score,
            is_anomalous=is_anomalous,
            threshold_applied=active_thresh,
        )
