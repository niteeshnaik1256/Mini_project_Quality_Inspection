"""
End-to-End Quality Inspection Pipeline.
Integrates ROI detection, anomaly detection, decision engine, temporal filtering, and event generation.
Provides a unified interface for the Quality Inspector role in the multi-role Smart Factory system.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.events.quality_event import QualityCVEvent, QualityEventLogger
from src.inference.inferencer import PatchCoreInferencer
from src.postprocessing.decision_engine import DecisionEngine, QualityDecision
from src.postprocessing.localizer import DefectLocalizer
from src.postprocessing.temporal_filter import TemporalFilter, TemporalFilterConfig
from src.roi.manual_roi import ManualROISelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QualityInspectionPipeline")


class QualityInspectionPipeline:
    """
    Complete quality inspection pipeline for automotive components.

    Pipeline stages:
    1. ROI extraction (manual or YOLO-based)
    2. Anomaly detection (PatchCore/EfficientAD/PaDiM)
    3. Decision engine (PASS/REVIEW/FAIL)
    4. Temporal validation (for video streams)
    5. Event generation (standardized CVEvent schema)
    6. Evidence saving (images, heatmaps)
    """

    def __init__(
        self,
        model_checkpoint: str | Path,
        decision_calibration_path: Optional[str | Path] = None,
        roi_config: Optional[Dict[str, Any]] = None,
        temporal_config: Optional[TemporalFilterConfig] = None,
        output_dir: str | Path = "results/inspections",
        enable_temporal: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize anomaly detection model
        logger.info(f"Loading anomaly detection model from: {model_checkpoint}")
        self.inferencer = PatchCoreInferencer(checkpoint_path=model_checkpoint)

        # Initialize decision engine
        if decision_calibration_path:
            logger.info(f"Loading decision calibration from: {decision_calibration_path}")
            self.decision_engine = DecisionEngine.load_calibration(decision_calibration_path)
        else:
            logger.warning("No decision calibration provided. Using default thresholds.")
            self.decision_engine = DecisionEngine()
            # Set conservative defaults
            self.decision_engine.set_thresholds(
                pass_threshold=1.0,
                fail_threshold=2.0,
                method="default",
            )

        # Initialize ROI selector
        self.roi_config = roi_config or {}
        self.roi_selector = ManualROISelector(config=self.roi_config)

        # Initialize defect localizer
        self.localizer = DefectLocalizer()

        # Initialize temporal filter (for video streams)
        self.enable_temporal = enable_temporal
        if enable_temporal:
            self.temporal_filter = TemporalFilter(config=temporal_config)
        else:
            self.temporal_filter = None

        # Initialize event logger
        self.event_logger = QualityEventLogger(self.output_dir / "events")

        # Create evidence directories
        self.evidence_dir = self.output_dir / "evidence"
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Quality Inspection Pipeline initialized successfully")

    def inspect_image(
        self,
        image: Union[str, Path, np.ndarray],
        camera_id: str = "default",
        product_id: str = "unknown",
        save_evidence: bool = True,
    ) -> QualityCVEvent:
        """
        Inspect a single image and generate a quality event.

        Args:
            image: Image path or numpy array
            camera_id: Camera identifier
            product_id: Product/component identifier
            save_evidence: Whether to save evidence images

        Returns:
            QualityCVEvent with inspection results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_array = cv2.imread(str(image_path))
            if image_array is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            image_array = image
            image_path = None

        # ROI extraction
        roi_crop, roi_metadata = self.roi_selector.extract_roi(image_array)

        # Anomaly detection
        inference_result = self.inferencer.predict(roi_crop)
        anomaly_score = inference_result["anomaly_score"]
        anomaly_map = inference_result["anomaly_map"]

        # Decision engine
        decision = self.decision_engine.decide(anomaly_score)

        # Defect localization
        bbox = self.localizer.extract_bbox(
            anomaly_map,
            threshold=self.decision_engine.thresholds.pass_threshold if self.decision_engine.thresholds else 0.5,
        )

        # Generate event
        event = QualityCVEvent.from_inference(
            camera_id=camera_id,
            product_id=product_id,
            anomaly_score=anomaly_score,
            decision=decision,
            bbox=bbox,
            metadata={
                "roi_metadata": roi_metadata,
                "inference_latency_ms": inference_result["latency_ms"],
                "device": inference_result["device"],
                "original_shape": inference_result["original_shape"],
            },
        )

        # Save evidence
        if save_evidence:
            evidence_filename = f"{event.event_id}_evidence.png"
            heatmap_filename = f"{event.event_id}_heatmap.png"

            evidence_path = self.evidence_dir / evidence_filename
            heatmap_path = self.heatmap_dir / heatmap_filename

            # Save evidence image with bbox overlay
            self._save_evidence_image(roi_crop, bbox, evidence_path)

            # Save heatmap
            self._save_heatmap(anomaly_map, heatmap_path)

            event.evidence_path = str(evidence_path)
            event.heatmap_path = str(heatmap_path)

        # Log event
        self.event_logger.log_event(event)

        logger.info(
            f"Inspection complete: {decision.value} (score: {anomaly_score:.4f}) "
            f"for product {product_id}"
        )

        return event

    def inspect_video_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        camera_id: str = "default",
        product_id: str = "unknown",
        save_evidence: bool = True,
    ) -> Tuple[Optional[QualityCVEvent], List[QualityCVEvent]]:
        """
        Inspect a video frame with temporal validation.

        Args:
            frame: Video frame as numpy array
            frame_idx: Frame number in video
            timestamp: Frame timestamp in seconds
            camera_id: Camera identifier
            product_id: Product identifier
            save_evidence: Whether to save evidence

        Returns:
            (new_event, events_to_emit): New event if triggered, and list of events to emit
        """
        if not self.enable_temporal:
            logger.warning("Temporal filtering not enabled. Use inspect_image for single frames.")
            event = self.inspect_image(frame, camera_id, product_id, save_evidence)
            return event, [event]

        # ROI extraction
        roi_crop, roi_metadata = self.roi_selector.extract_roi(frame)

        # Anomaly detection
        inference_result = self.inferencer.predict(roi_crop)
        anomaly_score = inference_result["anomaly_score"]
        anomaly_map = inference_result["anomaly_map"]

        # Decision engine
        decision = self.decision_engine.decide(anomaly_score)

        # Defect localization
        bbox = self.localizer.extract_bbox(
            anomaly_map,
            threshold=self.decision_engine.thresholds.pass_threshold if self.decision_engine.thresholds else 0.5,
        )

        # Temporal filtering
        temporal_event, _ = self.temporal_filter.process_frame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            anomaly_score=anomaly_score,
            decision=decision.value,
            bbox=bbox if bbox else None,
            camera_id=camera_id,
            metadata={
                "roi_metadata": roi_metadata,
                "inference_latency_ms": inference_result["latency_ms"],
            },
        )

        events_to_emit = []

        # If temporal event triggered, convert to CVEvent
        if temporal_event:
            # Save evidence for triggered events
            evidence_path = ""
            heatmap_path = ""

            if save_evidence:
                evidence_filename = f"{temporal_event.event_id}_evidence.png"
                heatmap_filename = f"{temporal_event.event_id}_heatmap.png"

                evidence_path = self.evidence_dir / evidence_filename
                heatmap_path = self.heatmap_dir / heatmap_filename

                self._save_evidence_image(roi_crop, bbox, evidence_path)
                self._save_heatmap(anomaly_map, heatmap_path)

            cv_event = QualityCVEvent.from_temporal_event(
                temporal_event=temporal_event,
                evidence_path=str(evidence_path),
                heatmap_path=str(heatmap_path),
                product_id=product_id,
            )

            self.event_logger.log_event(cv_event)
            events_to_emit.append(cv_event)

            logger.info(
                f"Temporal event triggered: {cv_event.decision} "
                f"(persistence: {len(temporal_event.anomaly_scores)} frames)"
            )

        return None, events_to_emit

    def _save_evidence_image(self, image: np.ndarray, bbox: List[int], filepath: Path) -> None:
        """Save evidence image with bounding box overlay."""
        vis_image = image.copy()

        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            color = (0, 0, 255)  # Red for defects
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

        cv2.imwrite(str(filepath), vis_image)

    def _save_heatmap(self, anomaly_map: np.ndarray, filepath: Path) -> None:
        """Save anomaly heatmap as color visualization."""
        # Normalize to 0-255
        heatmap_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        cv2.imwrite(str(filepath), heatmap_color)

    def get_inspection_stats(self) -> Dict[str, Any]:
        """Get statistics about inspections performed."""
        events = self.event_logger.load_events()

        decision_counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

        for event in events:
            decision_counts[event.decision] += 1
            severity_counts[event.severity] += 1

        return {
            "total_inspections": len(events),
            "decision_distribution": decision_counts,
            "severity_distribution": severity_counts,
        }
