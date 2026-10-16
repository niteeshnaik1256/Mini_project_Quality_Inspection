"""
Cross-model benchmark aggregation for PatchCore, EfficientAD, and PaDiM.
Builds comparison tables strictly from recorded evaluation JSON reports.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ModelComparison")

SUPPORTED_MODELS = ("patchcore", "efficientad", "padim")


@dataclass
class ModelBenchmarkRow:
    """Single row in the industrial model comparison table."""

    model: str
    image_auroc: Optional[float]
    f1_score: Optional[float]
    pixel_auroc: Optional[float]
    mean_latency_ms: Optional[float]
    fps: Optional[float]
    vram_usage_mb: Optional[float]
    metrics_path: Optional[str]
    missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "image_auroc": self.image_auroc,
            "f1_score": self.f1_score,
            "pixel_auroc": self.pixel_auroc,
            "mean_latency_ms": self.mean_latency_ms,
            "fps": self.fps,
            "vram_usage_mb": self.vram_usage_mb,
            "metrics_path": self.metrics_path,
            "missing": self.missing,
        }


def load_metrics_report(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def row_from_report(model: str, report: Dict[str, Any], metrics_path: Path) -> ModelBenchmarkRow:
    cls = report.get("classification_metrics", {})
    prod = report.get("production_metrics", {})
    return ModelBenchmarkRow(
        model=model,
        image_auroc=_float_or_none(report.get("image_auroc")),
        f1_score=_float_or_none(cls.get("f1_score")),
        pixel_auroc=_float_or_none(report.get("pixel_auroc")),
        mean_latency_ms=_float_or_none(prod.get("mean_latency_ms")),
        fps=_float_or_none(prod.get("fps")),
        vram_usage_mb=_float_or_none(prod.get("vram_usage_mb")),
        metrics_path=str(metrics_path.resolve()),
        missing=False,
    )


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_benchmark_rows(
    metrics_dir: Path,
    models: Optional[List[str]] = None,
) -> List[ModelBenchmarkRow]:
    """
    Load evaluation JSON files for each model.
    Missing files produce placeholder rows (missing=True) — no fabricated metrics.
    """
    target_models = list(models) if models else list(SUPPORTED_MODELS)
    rows: List[ModelBenchmarkRow] = []

    for model in target_models:
        metrics_path = metrics_dir / f"{model}_metrics.json"
        if not metrics_path.exists():
            logger.warning(f"No metrics file for '{model}': {metrics_path}")
            rows.append(
                ModelBenchmarkRow(
                    model=model,
                    image_auroc=None,
                    f1_score=None,
                    pixel_auroc=None,
                    mean_latency_ms=None,
                    fps=None,
                    vram_usage_mb=None,
                    metrics_path=None,
                    missing=True,
                )
            )
            continue

        report = load_metrics_report(metrics_path)
        rows.append(row_from_report(model, report, metrics_path))

    return rows


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def render_markdown_table(rows: List[ModelBenchmarkRow], title: str) -> str:
    """Render GitHub-flavored markdown comparison table."""
    lines = [
        f"# {title}",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "| Model | AUROC | F1 | Pixel AUROC | Latency (ms) | FPS | VRAM (MB) | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.missing:
            lines.append(
                f"| {row.model} | — | — | — | — | — | — | _not evaluated_ |"
            )
            continue
        source = Path(row.metrics_path or "").name
        lines.append(
            f"| {row.model} | {_fmt(row.image_auroc)} | {_fmt(row.f1_score)} | "
            f"{_fmt(row.pixel_auroc)} | {_fmt(row.mean_latency_ms, 2)} | "
            f"{_fmt(row.fps, 1)} | {_fmt(row.vram_usage_mb, 2)} | `{source}` |"
        )

    available = [r for r in rows if not r.missing]
    if len(available) >= 2:
        lines.extend(["", "## Quick read (experimental prototype only)", ""])
        best_auroc = max(available, key=lambda r: r.image_auroc or -1.0)
        best_pixel = max(available, key=lambda r: r.pixel_auroc or -1.0)
        fastest = min(available, key=lambda r: r.mean_latency_ms or float("inf"))
        lines.append(
            f"- **Highest image AUROC:** `{best_auroc.model}` ({_fmt(best_auroc.image_auroc)})"
        )
        lines.append(
            f"- **Highest pixel AUROC:** `{best_pixel.model}` ({_fmt(best_pixel.pixel_auroc)})"
        )
        lines.append(
            f"- **Lowest mean latency:** `{fastest.model}` ({_fmt(fastest.mean_latency_ms, 2)} ms)"
        )
        lines.append("")
        lines.append(
            "> Benchmarks reflect controlled AutoVI test-split experiments on development hardware; "
            "not real production-line validation."
        )

    lines.append("")
    return "\n".join(lines)


def write_csv_report(rows: List[ModelBenchmarkRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "image_auroc",
        "f1_score",
        "pixel_auroc",
        "mean_latency_ms",
        "fps",
        "vram_usage_mb",
        "metrics_path",
        "missing",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return output_path


def write_json_report(rows: List[ModelBenchmarkRow], output_path: Path, extra: Optional[Dict[str, Any]] = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [r.to_dict() for r in rows],
    }
    if extra:
        payload["context"] = extra
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path
