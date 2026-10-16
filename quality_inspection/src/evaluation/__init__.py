"""Evaluation and metrics module."""

from .comparison import collect_benchmark_rows, render_markdown_table
from .experiment_log import append_experiment_record
from .metrics import (
    compute_auroc,
    compute_classification_metrics,
    compute_pixel_auroc,
    compute_roc_curve,
    find_optimal_threshold,
)

__all__ = [
    "collect_benchmark_rows",
    "render_markdown_table",
    "append_experiment_record",
    "compute_auroc",
    "compute_pixel_auroc",
    "compute_classification_metrics",
    "compute_roc_curve",
    "find_optimal_threshold",
]


def __getattr__(name: str):
    if name == "ModelEvaluator":
        from .evaluator import ModelEvaluator

        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
