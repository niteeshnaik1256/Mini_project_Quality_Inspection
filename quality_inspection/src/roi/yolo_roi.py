"""
Optional YOLO-based ROI detection (component localization only — not defect classification).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.roi.config import ROIRectangle

logger = logging.getLogger("YOLOROI")


class YOLOROIDetector:
    """
    Detect product/component bounding boxes using Ultralytics YOLO weights.
    Falls back to full-frame manual rectangle when YOLO is unavailable or finds no boxes.
    """

    def __init__(
        self,
        weights_path: str,
        class_filter: Optional[int] = None,
        conf_threshold: float = 0.25,
        fallback_rectangle: Optional[ROIRectangle] = None,
    ):
        self.weights_path = weights_path
        self.class_filter = class_filter
        self.conf_threshold = conf_threshold
        self.fallback_rectangle = fallback_rectangle
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO ROI mode. Install with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.weights_path)
        logger.info("Loaded YOLO ROI detector weights: %s", self.weights_path)

    def detect_box(self, image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Return best detection box as (x1, y1, x2, y2) in pixel coordinates.
        """
        self._load_model()
        assert self._model is not None

        results = self._model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        if not results:
            return self._fallback_box(image_bgr)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback_box(image_bgr)

        best_conf = -1.0
        best_xyxy = None
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if self.class_filter is not None and cls_id != self.class_filter:
                continue
            if conf > best_conf:
                best_conf = conf
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                best_xyxy = tuple(xyxy)

        if best_xyxy is None:
            return self._fallback_box(image_bgr)
        return best_xyxy  # type: ignore[return-value]

    def _fallback_box(self, image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = image_bgr.shape[:2]
        if self.fallback_rectangle is not None:
            return self.fallback_rectangle.resolve(w, h)
        logger.warning("YOLO ROI found no detections; using full frame.")
        return 0, 0, w, h

    def crop(self, image_bgr: np.ndarray) -> tuple[np.ndarray, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self.detect_box(image_bgr)
        crop = image_bgr[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)
