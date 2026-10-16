"""
Industrial Anomaly Detection Metrics Engine.
Computes Image-level and Pixel-level AUROC, F1-Score, Precision, Recall,
Specificity, FPR, FNR, and optimal operating threshold determination.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MetricsEngine")


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute True Positive Rate (TPR) and False Positive Rate (FPR) across all score thresholds.
    Args:
        y_true: Binary ground truth labels (0 = nominal, 1 = anomalous)
        y_score: Continuous anomaly scores (higher indicates anomaly)
    Returns:
        fpr, tpr, thresholds
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Sort scores descending
    desc_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    # Distinct threshold points
    distinct_mask = np.diff(y_score_sorted, prepend=np.nan) != 0
    threshold_indices = np.where(distinct_mask)[0]

    # Cumulative true positives and false positives
    cum_tp = np.cumsum(y_true_sorted)
    cum_fp = np.cumsum(1 - y_true_sorted)

    total_tp = cum_tp[-1]
    total_fp = cum_fp[-1]

    if total_tp == 0:
        logger.warning("Zero positive samples in ground truth.")
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    if total_fp == 0:
        logger.warning("Zero negative samples in ground truth.")
        return np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    tpr = np.r_[0, cum_tp[threshold_indices] / total_tp, 1.0]
    fpr = np.r_[0, cum_fp[threshold_indices] / total_fp, 1.0]
    thresholds = np.r_[y_score_sorted[0] + 1e-4, y_score_sorted[threshold_indices], y_score_sorted[-1] - 1e-4]

    return fpr, tpr, thresholds


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute Area Under the Receiver Operating Characteristic (ROC) curve via trapezoidal rule."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_score)
    # Numerical integration: AUROC = \int TPR d(FPR)
    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return round(float(np.clip(auroc, 0.0, 1.0)), 4)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "f1",
) -> Tuple[float, float]:
    """
    Search for the optimal decision threshold.
    Methods:
      - 'f1': Maximizes the harmonic mean of precision and recall.
      - 'youden': Maximizes Youden's J statistic (Sensitivity + Specificity - 1 = TPR - FPR).
    Returns:
        (optimal_threshold, best_metric_value)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0

    best_thresh = float(thresholds[0])
    best_val = -1.0

    for thresh in thresholds:
        preds = (y_score >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        if method == "f1":
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_val:
                best_val = f1
                best_thresh = float(thresh)
        elif method == "youden":
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            j_stat = tpr - fpr
            if j_stat > best_val:
                best_val = j_stat
                best_thresh = float(thresh)

    return round(best_thresh, 4), round(best_val, 4)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute binary confusion matrix and industrial operating metrics at a given threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    preds = (y_score >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "accuracy": round(acc, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def compute_pixel_auroc(
    ground_truth_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    max_pixels: int = 250_000,
) -> float:
    """
    Compute Pixel-Level AUROC evaluating spatial defect localization fidelity.
    Uses uniform subsampling if pixel volume exceeds max_pixels for memory efficiency.
    """
    flat_masks = np.concatenate([m.ravel() for m in ground_truth_masks])
    flat_maps = np.concatenate([a.ravel() for a in anomaly_maps])

    binary_masks = (flat_masks > 0.5).astype(int)

    # Subsample if necessary for memory safety on consumer machines
    if len(binary_masks) > max_pixels:
        np.random.seed(42)
        indices = np.random.choice(len(binary_masks), size=max_pixels, replace=False)
        binary_masks = binary_masks[indices]
        flat_maps = flat_maps[indices]

    return compute_auroc(binary_masks, flat_maps)
