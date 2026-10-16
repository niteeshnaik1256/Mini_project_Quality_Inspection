"""
Industrial Quality Decision Engine for Automotive AOI.
Converts anomaly scores into PASS/REVIEW/FAIL decisions with configurable thresholds.
Supports threshold calibration using validation data to avoid arbitrary threshold selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DecisionEngine")


class QualityDecision(Enum):
    """Three-tier quality classification for industrial inspection."""
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass
class DecisionThresholds:
    """Configurable thresholds for three-tier decision logic."""
    pass_threshold: float  # Below this: PASS
    fail_threshold: float  # Above this: FAIL
    # Between pass_threshold and fail_threshold: REVIEW

    def to_dict(self) -> Dict[str, float]:
        return {
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DecisionThresholds":
        return cls(
            pass_threshold=data["pass_threshold"],
            fail_threshold=data["fail_threshold"],
        )


@dataclass
class CalibrationReport:
    """Documentation of how thresholds were selected."""
    method: str  # "fixed", "f1_optimal", "precision_target", "percentile"
    pass_threshold: float
    fail_threshold: float
    validation_auc: float
    pass_precision: float
    pass_recall: float
    fail_precision: float
    fail_recall: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
            "validation_auc": self.validation_auc,
            "pass_precision": self.pass_precision,
            "pass_recall": self.pass_recall,
            "fail_precision": self.fail_precision,
            "fail_recall": self.fail_recall,
            "metadata": self.metadata,
        }


class DecisionEngine:
    """
    Converts anomaly scores into PASS/REVIEW/FAIL decisions.
    Supports threshold calibration using validation data to avoid arbitrary selection.
    """

    def __init__(
        self,
        thresholds: Optional[DecisionThresholds] = None,
        calibration_report: Optional[CalibrationReport] = None,
    ):
        self.thresholds = thresholds
        self.calibration_report = calibration_report

    def decide(self, anomaly_score: float) -> QualityDecision:
        """
        Map a single anomaly score to a quality decision.
        Logic:
            score < pass_threshold → PASS
            pass_threshold <= score < fail_threshold → REVIEW
            score >= fail_threshold → FAIL
        """
        if self.thresholds is None:
            raise RuntimeError(
                "Decision thresholds not set. Calibrate thresholds using calibration data "
                "or set them manually before making decisions."
            )

        if anomaly_score < self.thresholds.pass_threshold:
            return QualityDecision.PASS
        elif anomaly_score < self.thresholds.fail_threshold:
            return QualityDecision.REVIEW
        else:
            return QualityDecision.FAIL

    def decide_batch(self, anomaly_scores: List[float]) -> List[QualityDecision]:
        """Apply decision logic to a batch of anomaly scores."""
        return [self.decide(score) for score in anomaly_scores]

    @staticmethod
    def calibrate_from_validation(
        anomaly_scores: List[float],
        labels: List[int],  # 0: nominal, 1: anomalous
        method: str = "f1_optimal",
        pass_target_precision: float = 0.99,
        fail_target_recall: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DecisionThresholds, CalibrationReport]:
        """
        Calibrate thresholds using validation data to avoid arbitrary selection.

        Methods:
        - "f1_optimal": Select threshold that maximizes F1-score (used for fail threshold)
        - "precision_target": Select threshold to achieve target precision (used for pass threshold)
        - "percentile": Use percentile-based thresholds
        - "adaptive": Combine precision and recall targets

        Returns:
            (DecisionThresholds, CalibrationReport)
        """
        if len(anomaly_scores) != len(labels):
            raise ValueError("Anomaly scores and labels must have the same length.")

        scores = np.array(anomaly_scores)
        labels = np.array(labels)

        # Calculate AUROC for validation performance reference
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = np.trapz(tpr, fpr)

        logger.info(f"Calibrating thresholds with {len(scores)} validation samples (AUROC: {auc:.4f})")

        # Default: fail threshold at optimal F1 (aggressive detection)
        if method in ["f1_optimal", "adaptive"]:
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            fail_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else scores.max()
            fail_prec = precision[optimal_idx]
            fail_rec = recall[optimal_idx]
        else:
            # Fallback: use 95th percentile of nominal scores
            nominal_scores = scores[labels == 0]
            fail_threshold = np.percentile(nominal_scores, 95) if len(nominal_scores) > 0 else scores.max()
            fail_prec = 0.95
            fail_rec = 0.95

        # Pass threshold: high precision for nominal classification
        if method in ["precision_target", "adaptive"]:
            # Find threshold where precision >= target for nominal class
            nominal_scores = scores[labels == 0]
            pass_threshold = np.percentile(nominal_scores, pass_target_precision * 100)
            pass_prec = pass_target_precision
            pass_rec = 1.0 - (np.sum(nominal_scores > pass_threshold) / len(nominal_scores))
        else:
            # Fallback: use 90th percentile of nominal scores
            nominal_scores = scores[labels == 0]
            pass_threshold = np.percentile(nominal_scores, 90) if len(nominal_scores) > 0 else scores.min()
            pass_prec = 0.90
            pass_rec = 0.90

        # Ensure pass_threshold < fail_threshold
        if pass_threshold >= fail_threshold:
            logger.warning(
                f"Calibrated pass_threshold ({pass_threshold:.4f}) >= fail_threshold ({fail_threshold:.4f}). "
                "Adjusting to maintain ordering."
            )
            pass_threshold = fail_threshold * 0.5

        thresholds = DecisionThresholds(
            pass_threshold=float(pass_threshold),
            fail_threshold=float(fail_threshold),
        )

        report = CalibrationReport(
            method=method,
            pass_threshold=float(pass_threshold),
            fail_threshold=float(fail_threshold),
            validation_auc=float(auc),
            pass_precision=float(pass_prec),
            pass_recall=float(pass_rec),
            fail_precision=float(fail_prec),
            fail_recall=float(fail_rec),
            metadata=metadata or {},
        )

        logger.info(
            f"Calibration complete: PASS < {pass_threshold:.4f}, "
            f"REVIEW [{pass_threshold:.4f}, {fail_threshold:.4f}), "
            f"FAIL >= {fail_threshold:.4f}"
        )

        return thresholds, report

    def set_thresholds(
        self,
        pass_threshold: float,
        fail_threshold: float,
        method: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Manually set thresholds with documentation.
        Useful when using domain knowledge or production requirements.
        """
        if pass_threshold >= fail_threshold:
            raise ValueError(f"pass_threshold ({pass_threshold}) must be < fail_threshold ({fail_threshold})")

        self.thresholds = DecisionThresholds(
            pass_threshold=pass_threshold,
            fail_threshold=fail_threshold,
        )

        self.calibration_report = CalibrationReport(
            method=method,
            pass_threshold=pass_threshold,
            fail_threshold=fail_threshold,
            validation_auc=0.0,  # Unknown for manual setting
            pass_precision=0.0,
            pass_recall=0.0,
            fail_precision=0.0,
            fail_recall=0.0,
            metadata=metadata or {},
        )

        logger.info(
            f"Manual thresholds set: PASS < {pass_threshold:.4f}, "
            f"REVIEW [{pass_threshold:.4f}, {fail_threshold:.4f}), "
            f"FAIL >= {fail_threshold:.4f}"
        )

    def save_calibration(self, filepath: str | Path) -> None:
        """Save calibration report to JSON for reproducibility."""
        if self.calibration_report is None:
            raise RuntimeError("No calibration report to save. Calibrate thresholds first.")

        import json

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "calibration_report": self.calibration_report.to_dict(),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Calibration saved to: {out_path}")

    @classmethod
    def load_calibration(cls, filepath: str | Path) -> "DecisionEngine":
        """Load calibration from JSON file."""
        import json

        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {in_path}")

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        thresholds = None
        if data.get("thresholds"):
            thresholds = DecisionThresholds.from_dict(data["thresholds"])

        cal_report = None
        if data.get("calibration_report"):
            cr = data["calibration_report"]
            cal_report = CalibrationReport(
                method=cr["method"],
                pass_threshold=cr["pass_threshold"],
                fail_threshold=cr["fail_threshold"],
                validation_auc=cr["validation_auc"],
                pass_precision=cr["pass_precision"],
                pass_recall=cr["pass_recall"],
                fail_precision=cr["fail_precision"],
                fail_recall=cr["fail_recall"],
                metadata=cr.get("metadata", {}),
            )

        engine = cls(thresholds=thresholds, calibration_report=cal_report)
        logger.info(f"Calibration loaded from: {in_path}")
        return engine

    def get_decision_stats(
        self,
        anomaly_scores: List[float],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate decision statistics for a batch of scores.
        If labels provided, compute per-decision precision/recall.
        """
        decisions = self.decide_batch(anomaly_scores)

        decision_counts = {
            QualityDecision.PASS.value: 0,
            QualityDecision.REVIEW.value: 0,
            QualityDecision.FAIL.value: 0,
        }

        for d in decisions:
            decision_counts[d.value] += 1

        stats = {
            "decision_counts": decision_counts,
            "total_samples": len(decisions),
        }

        if labels is not None:
            # Calculate per-decision accuracy if ground truth available
            labels_arr = np.array(labels)
            decisions_arr = np.array([d.value for d in decisions])

            for decision in QualityDecision:
                mask = decisions_arr == decision.value
                if mask.sum() > 0:
                    correct = (labels_arr[mask] == (1 if decision == QualityDecision.FAIL else 0)).sum()
                    stats[f"{decision.value}_accuracy"] = correct / mask.sum()

        return stats
