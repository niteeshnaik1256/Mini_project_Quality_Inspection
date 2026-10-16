"""
Unified inference benchmark for PatchCore, EfficientAD, and PaDiM.
Reports latency, FPS, and optional VRAM on representative test partitions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inferencer import PatchCoreInferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchmarkCLI")

DEFAULT_CHECKPOINTS = {
    "patchcore": PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt",
    "efficientad": PROJECT_ROOT / "models" / "efficientad" / "efficientad_checkpoint.pt",
    "padim": PROJECT_ROOT / "models" / "padim" / "padim_checkpoint.pt",
}


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection inference")
    parser.add_argument(
        "--model",
        type=str,
        default="padim",
        choices=["patchcore", "efficientad", "padim"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly" / "test"),
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.model]
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Train the model first.")
        sys.exit(1)

    test_root = Path(args.data_dir)
    if not test_root.exists():
        logger.error(f"Test directory not found: {test_root}")
        sys.exit(1)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inferencer = PatchCoreInferencer(checkpoint_path=ckpt_path)
    sample_paths = []
    for folder in sorted(test_root.iterdir()):
        if folder.is_dir():
            imgs = sorted(folder.glob("*.png"))
            if imgs:
                sample_paths.append(imgs[0])

    if not sample_paths:
        logger.error(f"No PNG samples found under {test_root}")
        sys.exit(1)

    for _ in range(args.warmup):
        inferencer.predict(sample_paths[0])

    latencies = []
    for img_path in sample_paths:
        res = inferencer.predict(img_path)
        latencies.append(res["latency_ms"])

    mean_lat = sum(latencies) / len(latencies)
    fps = 1000.0 / max(mean_lat, 1e-6)

    print("\n" + "=" * 70)
    print(f"  INFERENCE BENCHMARK — {args.model.upper()}")
    print("=" * 70)
    print(f"Checkpoint:        {ckpt_path}")
    print(f"Samples timed:     {len(sample_paths)}")
    print(f"Mean latency:      {mean_lat:.2f} ms")
    print(f"Throughput (FPS):  {fps:.1f}")
    print(f"Peak VRAM:         {vram_mb():.1f} MB")
    print(f"Device:            {inferencer.device.upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
