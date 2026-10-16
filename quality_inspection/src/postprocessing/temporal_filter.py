"""
Temporal Validation for Video-Stream Quality Inspection.
Avoids generating a new quality alert on every frame by requiring persistence.
Implements frame-level persistence and duplicate-event suppression for real-time AOI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TemporalFilter")


class EventType(Enum):
    """Event types for quality inspection."""
    QUALITY_ANOMALY = "quality_anomaly"
    SYSTEM_EVENT = "system_event"


@dataclass
class TemporalEvent:
    """A persistent quality event spanning multiple frames."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    camera_id: str = "default"
    event_type: EventType = EventType.QUALITY_ANOMALY
    start_frame: int = 0
    end_frame: Optional[int] = None
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    decision: str = "UNKNOWN"  # PASS, REVIEW, FAIL
    severity: str = "unknown"
    confidence: float = 0.0
    anomaly_scores: List[float] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "event_type": self.event_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "first_seen_timestamp": self.first_seen_timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "decision": self.decision,
            "severity": self.severity,
            "confidence": self.confidence,
            "anomaly_scores": self.anomaly_scores,
            "bboxes": self.bboxes,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class TemporalFilterConfig:
    """Configuration for temporal validation logic."""
    minimum_persistence_frames: int = 3  # Minimum consecutive frames to trigger event
    max_cooldown_frames: int = 30  # Frames before duplicate event can re-occur
    spatial_similarity_threshold: float = 0.5  # IoU threshold for event matching
    temporal_window_seconds: float = 2.0  # Time window for event consolidation
    max_event_duration_frames: int = 300  # Max frames before event auto-closes


class TemporalFilter:
    """
    Temporal validation for video-stream quality inspection.
    Implements:
    1. Frame-level persistence (minimum_persistence_frames)
    2. Duplicate-event suppression (spatial + temporal)
    3. Event lifecycle management (start, update, end)
    """

    def __init__(self, config: Optional[TemporalFilterConfig] = None):
        self.config = config or TemporalFilterConfig()
        self.active_events: List[TemporalEvent] = []
        self.closed_events: List[TemporalEvent] = []
        self.frame_counter = 0
        self.event_cooldowns: Dict[str, int] = {}  # event_id -> remaining cooldown frames

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        anomaly_score: float,
        decision: str,
        bbox: Optional[List[int]] = None,
        camera_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[TemporalEvent], List[TemporalEvent]]:
        """
        Process a single frame through temporal validation.

        Args:
            frame_idx: Current frame number
            timestamp: Frame timestamp in seconds
            anomaly_score: Anomaly score from detector
            decision: PASS/REVIEW/FAIL decision
            bbox: Bounding box [x, y, w, h] if available
            camera_id: Camera identifier
            metadata: Additional metadata

        Returns:
            (new_event, events_to_emit): New event if triggered, and list of events to emit
        """
        self.frame_counter = frame_idx

        # Update cooldowns
        self._update_cooldowns()

        # Check if this is a quality-relevant decision
        if decision == "PASS":
            return None, []

        # Find matching active event
        matched_event = self._find_matching_event(bbox, camera_id)

        if matched_event:
            # Update existing event
            self._update_event(matched_event, frame_idx, timestamp, anomaly_score, decision, bbox, metadata)
            return None, []
        else:
            # Check if this is in cooldown (duplicate suppression)
            if self._is_in_cooldown(camera_id, bbox):
                logger.debug(f"Frame {frame_idx}: Suppressed duplicate event (cooldown)")
                return None, []

            # Start new potential event
            new_event = self._start_event(
                frame_idx, timestamp, anomaly_score, decision, bbox, camera_id, metadata
            )
            self.active_events.append(new_event)

            # Check if event meets persistence threshold
            if len(new_event.anomaly_scores) >= self.config.minimum_persistence_frames:
                logger.info(
                    f"Frame {frame_idx}: Event {new_event.event_id} triggered "
                    f"(persistence: {len(new_event.anomaly_scores)} frames)"
                )
                # Start cooldown for this location
                self._start_cooldown(camera_id, bbox, new_event.event_id)
                return new_event, []
            else:
                logger.debug(
                    f"Frame {frame_idx}: Event {new_event.event_id} pending "
                    f"(persistence: {len(new_event.anomaly_scores)}/{self.config.minimum_persistence_frames})"
                )
                return None, []

    def _update_cooldowns(self) -> None:
        """Decrement cooldown counters."""
        for event_id in list(self.event_cooldowns.keys()):
            self.event_cooldowns[event_id] -= 1
            if self.event_cooldowns[event_id] <= 0:
                del self.event_cooldowns[event_id]

    def _find_matching_event(
        self,
        bbox: Optional[List[int]],
        camera_id: str,
    ) -> Optional[TemporalEvent]:
        """Find active event that spatially matches current detection."""
        if bbox is None:
            # No spatial info: match any active event for this camera
            for event in self.active_events:
                if event.camera_id == camera_id and event.is_active:
                    return event
            return None

        # With spatial info: use IoU matching
        for event in self.active_events:
            if event.camera_id == camera_id and event.is_active and event.bboxes:
                # Check against latest bbox
                latest_bbox = event.bboxes[-1]
                iou = self._calculate_iou(bbox, latest_bbox)
                if iou >= self.config.spatial_similarity_threshold:
                    return event

        return None

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / max(union_area, 1e-6)

    def _update_event(
        self,
        event: TemporalEvent,
        frame_idx: int,
        timestamp: float,
        anomaly_score: float,
        decision: str,
        bbox: Optional[List[int]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Update existing event with new frame data."""
        event.end_frame = frame_idx
        event.last_seen_timestamp = timestamp
        event.anomaly_scores.append(anomaly_score)
        event.decision = decision  # Update to latest decision
        event.confidence = max(event.confidence, anomaly_score)

        if bbox:
            event.bboxes.append(bbox)

        if metadata:
            event.metadata.update(metadata)

        # Check for max duration
        if frame_idx - event.start_frame >= self.config.max_event_duration_frames:
            logger.info(f"Event {event.event_id} reached max duration, closing")
            self._close_event(event)

    def _start_event(
        self,
        frame_idx: int,
        timestamp: float,
        anomaly_score: float,
        decision: str,
        bbox: Optional[List[int]],
        camera_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> TemporalEvent:
        """Create new event."""
        event = TemporalEvent(
            camera_id=camera_id,
            start_frame=frame_idx,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            decision=decision,
            confidence=anomaly_score,
            anomaly_scores=[anomaly_score],
            bboxes=[bbox] if bbox else [],
            metadata=metadata or {},
        )

        # Set severity based on decision
        if decision == "FAIL":
            event.severity = "high"
        elif decision == "REVIEW":
            event.severity = "medium"
        else:
            event.severity = "low"

        return event

    def _close_event(self, event: TemporalEvent) -> None:
        """Close an active event."""
        event.is_active = False
        if event in self.active_events:
            self.active_events.remove(event)
        self.closed_events.append(event)
        logger.info(f"Closed event {event.event_id} (duration: {event.end_frame - event.start_frame} frames)")

    def _is_in_cooldown(self, camera_id: str, bbox: Optional[List[int]]) -> bool:
        """Check if a similar event is in cooldown period."""
        if not self.event_cooldowns:
            return False

        # Simple camera-level cooldown if no spatial info
        if bbox is None:
            return any(self.event_cooldowns.values())

        # Spatial-aware cooldown
        for event_id, cooldown_frames in self.event_cooldowns.items():
            if cooldown_frames > 0:
                # Find the closed event
                closed_event = next((e for e in self.closed_events if e.event_id == event_id), None)
                if closed_event and closed_event.bboxes:
                    latest_bbox = closed_event.bboxes[-1]
                    iou = self._calculate_iou(bbox, latest_bbox)
                    if iou >= self.config.spatial_similarity_threshold:
                        return True

        return False

    def _start_cooldown(self, camera_id: str, bbox: Optional[List[int]], event_id: str) -> None:
        """Start cooldown period for an event location."""
        self.event_cooldowns[event_id] = self.config.max_cooldown_frames
        logger.debug(f"Started cooldown for event {event_id} ({self.config.max_cooldown_frames} frames)")

    def force_close_inactive_events(self, current_frame: int, timestamp: float) -> List[TemporalEvent]:
        """
        Close events that haven't been updated recently.
        Returns list of newly closed events.
        """
        newly_closed = []
        inactive_threshold = self.config.max_cooldown_frames

        for event in list(self.active_events):
            frames_since_update = current_frame - event.end_frame if event.end_frame else 0
            if frames_since_update > inactive_threshold:
                logger.info(f"Force-closing inactive event {event.event_id}")
                self._close_event(event)
                newly_closed.append(event)

        return newly_closed

    def get_active_events(self) -> List[TemporalEvent]:
        """Return list of currently active events."""
        return [e for e in self.active_events if e.is_active]

    def get_closed_events(self) -> List[TemporalEvent]:
        """Return list of closed events."""
        return self.closed_events.copy()

    def reset(self) -> None:
        """Reset filter state (clear all events)."""
        self.active_events.clear()
        self.closed_events.clear()
        self.event_cooldowns.clear()
        self.frame_counter = 0
        logger.info("Temporal filter reset")
