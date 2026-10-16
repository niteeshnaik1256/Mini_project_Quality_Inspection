"""
Standardized CVEvent Schema for Quality Inspector.
Compatible with Safety Officer and Production Manager roles in the multi-role system.
Implements the common event schema specification for Smart Factory Monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.postprocessing.decision_engine import QualityDecision
from src.postprocessing.temporal_filter import TemporalEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QualityEvent")


@dataclass
class QualityCVEvent:
    """
    Standardized CVEvent for Quality Inspector role.
    Schema compatible with multi-role Smart Factory Monitoring system.
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    camera_id: str = "default"
    role: str = "quality"
    task: str = "anomaly_inspection"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    product_id: str = "unknown"

    # Quality-specific fields
    anomaly_score: float = 0.0
    decision: str = "PASS"  # PASS, REVIEW, FAIL
    severity: str = "unknown"  # low, medium, high

    # Localization
    bbox: List[int] = field(default_factory=list)  # [x, y, w, h]
    track_id: Optional[str] = None

    # Evidence paths
    evidence_path: str = ""
    heatmap_path: str = ""

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "role": self.role,
            "task": self.task,
            "timestamp": self.timestamp,
            "product_id": self.product_id,
            "anomaly_score": self.anomaly_score,
            "decision": self.decision,
            "severity": self.severity,
            "bbox": self.bbox,
            "track_id": self.track_id,
            "evidence_path": self.evidence_path,
            "heatmap_path": self.heatmap_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_temporal_event(
        cls,
        temporal_event: TemporalEvent,
        evidence_path: str = "",
        heatmap_path: str = "",
        product_id: str = "unknown",
    ) -> "QualityCVEvent":
        """
        Create QualityCVEvent from TemporalEvent.
        Converts internal temporal event to standardized schema.
        """
        # Use average anomaly score
        avg_score = (
            sum(temporal_event.anomaly_scores) / len(temporal_event.anomaly_scores)
            if temporal_event.anomaly_scores
            else 0.0
        )

        # Use latest bbox if available
        bbox = temporal_event.bboxes[-1] if temporal_event.bboxes else []

        event = cls(
            event_id=temporal_event.event_id,
            camera_id=temporal_event.camera_id,
            decision=temporal_event.decision,
            severity=temporal_event.severity,
            anomaly_score=avg_score,
            bbox=bbox,
            track_id=None,  # Can be added if tracking is implemented
            evidence_path=evidence_path,
            heatmap_path=heatmap_path,
            product_id=product_id,
            metadata={
                "start_frame": temporal_event.start_frame,
                "end_frame": temporal_event.end_frame,
                "first_seen_timestamp": temporal_event.first_seen_timestamp,
                "last_seen_timestamp": temporal_event.last_seen_timestamp,
                "persistence_frames": len(temporal_event.anomaly_scores),
                "event_type": temporal_event.event_type.value,
                **temporal_event.metadata,
            },
        )

        return event

    @classmethod
    def from_inference(
        cls,
        camera_id: str,
        product_id: str,
        anomaly_score: float,
        decision: QualityDecision,
        bbox: Optional[List[int]] = None,
        evidence_path: str = "",
        heatmap_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QualityCVEvent":
        """
        Create QualityCVEvent from single-frame inference result.
        Used for non-temporal (single-image) inspection.
        """
        severity_map = {
            QualityDecision.PASS: "low",
            QualityDecision.REVIEW: "medium",
            QualityDecision.FAIL: "high",
        }

        event = cls(
            camera_id=camera_id,
            product_id=product_id,
            anomaly_score=anomaly_score,
            decision=decision.value,
            severity=severity_map.get(decision, "unknown"),
            bbox=bbox or [],
            evidence_path=evidence_path,
            heatmap_path=heatmap_path,
            metadata=metadata or {},
        )

        return event

    def validate(self) -> bool:
        """
        Validate event against schema requirements.
        Returns True if valid, False otherwise.
        """
        required_fields = [
            "event_id",
            "camera_id",
            "role",
            "task",
            "timestamp",
            "anomaly_score",
            "decision",
        ]

        event_dict = self.to_dict()

        for field in required_fields:
            if field not in event_dict or event_dict[field] is None:
                logger.error(f"Validation failed: missing required field '{field}'")
                return False

        # Validate decision value
        if self.decision not in ["PASS", "REVIEW", "FAIL"]:
            logger.error(f"Validation failed: invalid decision '{self.decision}'")
            return False

        # Validate severity value
        if self.severity not in ["low", "medium", "high", "unknown"]:
            logger.error(f"Validation failed: invalid severity '{self.severity}'")
            return False

        # Validate timestamp format
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError:
            logger.error(f"Validation failed: invalid timestamp format '{self.timestamp}'")
            return False

        return True

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json

        if not self.validate():
            raise ValueError("Cannot serialize invalid event to JSON")

        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "QualityCVEvent":
        """Create QualityCVEvent from JSON string."""
        import json

        data = json.loads(json_str)

        return cls(
            event_id=data["event_id"],
            camera_id=data["camera_id"],
            role=data["role"],
            task=data["task"],
            timestamp=data["timestamp"],
            product_id=data["product_id"],
            anomaly_score=data["anomaly_score"],
            decision=data["decision"],
            severity=data["severity"],
            bbox=data.get("bbox", []),
            track_id=data.get("track_id"),
            evidence_path=data.get("evidence_path", ""),
            heatmap_path=data.get("heatmap_path", ""),
            metadata=data.get("metadata", {}),
        )

    def save(self, filepath: str | Path) -> None:
        """Save event to JSON file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        logger.info(f"Event saved to: {out_path}")

    @classmethod
    def load(cls, filepath: str | Path) -> "QualityCVEvent":
        """Load event from JSON file."""
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Event file not found: {in_path}")

        with open(in_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


class QualityEventLogger:
    """
    Logger for QualityCVEvents.
    Handles event serialization, storage, and compatibility with multi-role system.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.output_dir / "quality_events.jsonl"

    def log_event(self, event: QualityCVEvent) -> None:
        """Append event to JSONL log file."""
        if not event.validate():
            logger.error(f"Cannot log invalid event: {event.event_id}")
            return

        with open(self.event_log_path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

        logger.info(f"Logged event {event.event_id} to {self.event_log_path}")

    def log_events_batch(self, events: List[QualityCVEvent]) -> None:
        """Log multiple events in batch."""
        for event in events:
            self.log_event(event)

    def load_events(self, limit: Optional[int] = None) -> List[QualityCVEvent]:
        """Load events from log file."""
        if not self.event_log_path.exists():
            return []

        events = []
        with open(self.event_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(QualityCVEvent.from_json(line))
                    if limit and len(events) >= limit:
                        break

        return events

    def clear_log(self) -> None:
        """Clear event log file."""
        if self.event_log_path.exists():
            self.event_log_path.unlink()
            logger.info(f"Cleared event log: {self.event_log_path}")
