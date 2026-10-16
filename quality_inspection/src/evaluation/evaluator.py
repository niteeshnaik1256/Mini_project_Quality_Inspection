"""
Comprehensive Model Evaluation & Benchmarking Harness.
Executes batch evaluation over test splits, records image & pixel metrics,
profiles latency/memory, and generates ROC curves and audit JSON logs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from src.data.dataset import AutomotiveAnomalyDataset
from src.evaluation.metrics import (
    compute_auroc,
    compute_classification_metrics,
    compute_pixel_auroc,
    compute_roc_curve,
    find_optimal_threshold,
)
from src.inference.inferencer import PatchCoreInferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelEvaluator")


class ModelEvaluator:
    """
    Standardized benchmark harness for industrial quality inspection models.
    Ensures identical evaluation protocol across PatchCore, EfficientAD, and PaDiM.
    """

    def __init__(
        self,
        inferencer: PatchCoreInferencer,
        test_dataset: AutomotiveAnomalyDataset,
        model_name: str = "patchcore",
    ):
        self.inferencer = inferencer
        self.test_dataset = test_dataset
        self.model_name = model_name

    def evaluate(self) -> Dict[str, Any]:
        """
        Execute full test evaluation.
        Computes Image AUROC, Pixel AUROC, F1, Latency, and Memory Footprint.
        """
        logger.info(f"Starting evaluation of model: '{self.model_name}' on {len(self.test_dataset)} test samples...")

        y_true_labels: List[int] = []
        y_pred_scores: List[float] = []
        gt_masks: List[np.ndarray] = []
        pred_maps: List[np.ndarray] = []
        latencies_ms: List[float] = []

        initial_process = psutil.Process()
        cpu_samples: List[float] = []

        for idx in range(len(self.test_dataset)):
            sample = self.test_dataset[idx]
            img_path = sample["image_path"]
            label = int(sample["label"])
            mask_arr = sample["mask"].squeeze(0).numpy()

            # Measure inference latency
            t0 = time.perf_counter()
            res = self.inferencer.predict(img_path)
            lat = (time.perf_counter() - t0) * 1000.0

            latencies_ms.append(lat)
            y_true_labels.append(label)
            y_pred_scores.append(res["anomaly_score"])
            gt_masks.append(mask_arr)
            pred_maps.append(res["anomaly_map"])

            cpu_samples.append(psutil.cpu_percent(interval=None))

        y_true_arr = np.array(y_true_labels, dtype=int)
        y_score_arr = np.array(y_pred_scores, dtype=float)

        # 1. Image-Level Metrics
        image_auroc = compute_auroc(y_true_arr, y_score_arr)
        opt_thresh, max_f1 = find_optimal_threshold(y_true_arr, y_score_arr, method="f1")
        class_metrics = compute_classification_metrics(y_true_arr, y_score_arr, threshold=opt_thresh)

        # 2. Pixel-Level Localization Metrics
        pixel_auroc = compute_pixel_auroc(gt_masks, pred_maps)

        # 3. Production Latency & System Resources
        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        fps = float(1000.0 / max(mean_lat, 1e-4))
        ram_mb = float(initial_process.memory_info().rss / (1024 * 1024))
        avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

        report = {
            "model": self.model_name,
            "samples_evaluated": len(self.test_dataset),
            "nominal_samples": int(np.sum(y_true_arr == 0)),
            "anomalous_samples": int(np.sum(y_true_arr == 1)),
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "optimal_threshold": opt_thresh,
            "classification_metrics": class_metrics,
            "production_metrics": {
                "mean_latency_ms": round(mean_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "fps": round(fps, 1),
                "ram_usage_mb": round(ram_mb, 2),
                "vram_usage_mb": round(vram_mb, 2),
                "avg_cpu_percent": round(avg_cpu, 1),
                "device": self.inferencer.device,
            },
        }

        return report

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """Render and save Receiver Operating Characteristic (ROC) curve."""
        fpr, tpr, _ = compute_roc_curve(y_true, y_score)
        auroc = compute_auroc(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{self.model_name.upper()} (AUROC = {auroc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
        ax.set_title(f"ROC Curve — {self.model_name.upper()} Evaluation", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.4)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curve saved to: {out}")
        return out

    def save_report(self, report: Dict[str, Any], output_path: str | Path) -> None:
        """Serialize structured metrics report to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Metrics report written to: {out}")
