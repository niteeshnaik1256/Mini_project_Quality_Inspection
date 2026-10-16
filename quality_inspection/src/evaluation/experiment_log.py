"""
Structured experiment tracking (JSON Lines) for reproducible AOI benchmarks.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch

    _TORCH_VERSION = torch.__version__
except ImportError:
    _TORCH_VERSION = "not_installed"


def append_experiment_record(
    log_path: Path,
    report: Dict[str, Any],
    *,
    dataset_version: str = "autovi",
    category: str = "bracket_assembly",
    backbone: Optional[str] = None,
    image_size: Optional[Any] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    training_config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Path:
    """
    Append one evaluation run to a JSONL experiment log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "category": category,
        "model": report.get("model"),
        "backbone": backbone,
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": seed,
        "training_config": training_config_path,
        "checkpoint": checkpoint_path,
        "threshold": report.get("optimal_threshold"),
        "metrics": {
            "image_auroc": report.get("image_auroc"),
            "pixel_auroc": report.get("pixel_auroc"),
            "classification": report.get("classification_metrics"),
            "production": report.get("production_metrics"),
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": _TORCH_VERSION,
            "device": report.get("production_metrics", {}).get("device"),
        },
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return log_path
