"""
PatchCore Baseline Training Script.
Reads YAML configuration, processes nominal training images, executes greedy coreset reduction,
and saves the trained memory bank checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders
from src.models.patchcore import PatchCoreModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainPatchCore")


def set_seed(seed: int) -> None:
    """Set global deterministic random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore Industrial Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "patchcore.yaml"),
        help="Path to PatchCore configuration YAML",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
        help="Override coreset sampling ratio (e.g. 0.10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoint",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides if present
    if args.coreset_ratio is not None:
        config["model"]["coreset_sampling_ratio"] = args.coreset_ratio
    if args.output_dir is not None:
        config["trainer"]["output_dir"] = args.output_dir

    seed = config["trainer"].get("seed", 42)
    set_seed(seed)

    logger.info("=" * 65)
    logger.info("   INDUSTRIAL AOI: PATCHCORE BASELINE TRAINING")
    logger.info("=" * 65)
    logger.info(f"Target Category:        {config['dataset']['category']}")
    logger.info(f"Backbone:               {config['model']['backbone']}")
    logger.info(f"Feature Layers:         {config['model']['layers']}")
    logger.info(f"Coreset Sampling Ratio: {config['model']['coreset_sampling_ratio']:.1%}")
    logger.info(f"Image Resolution:       {config['dataset']['image_size']}")
    logger.info("-" * 65)

    # 1. Build DataLoaders
    train_loader, test_loader = build_dataloaders(config)
    logger.info(f"Nominal training batches: {len(train_loader)} (Batch size: {config['dataset']['train_batch_size']})")

    # 2. Instantiate PatchCore Model
    device = "cuda" if torch.cuda.is_available() and config["trainer"]["device"] == "auto" else "cpu"
    logger.info(f"Compute Device:         {device.upper()}")

    model = PatchCoreModel(
        backbone_name=config["model"]["backbone"],
        layers=tuple(config["model"]["layers"]),
        patch_size=config["model"].get("patch_size", 3),
        patch_stride=config["model"].get("patch_stride", 1),
        coreset_sampling_ratio=config["model"]["coreset_sampling_ratio"],
        num_neighbors=config["model"].get("num_neighbors", 9),
        dimension_reduction=config["model"].get("dimension_reduction", 128),
        device=device,
    )

    # 3. Fit Memory Bank & Minimax Coreset Subsampling
    start_time = time.time()
    model.fit(train_loader)
    elapsed_time = time.time() - start_time

    # 4. Save Checkpoint
    out_dir = Path(config["trainer"]["output_dir"])
    ckpt_name = config["trainer"].get("checkpoint_name", "patchcore_checkpoint.pt")
    ckpt_path = out_dir / ckpt_name

    meta = {
        "category": config["dataset"]["category"],
        "training_samples": len(train_loader.dataset),
        "fit_time_seconds": round(elapsed_time, 2),
        "coreset_size": model.memory_bank.shape[0] if model.memory_bank is not None else 0,
        "feature_dim": model.embedding_dimension,
    }
    saved_path = model.save_checkpoint(ckpt_path, extra_metadata=meta)

    # 5. Quick Smoke Test on 1 Test Sample
    logger.info("Executing smoke-test inference on single test sample...")
    test_sample = next(iter(test_loader))
    sample_img = test_sample["image"][0:1]  # (1, 3, H, W)
    res = model.predict_step(sample_img)

    logger.info("-" * 65)
    logger.info("TRAINING AUDIT & BENCHMARK SUMMARY:")
    logger.info(f"  * Elapsed Training Time: {elapsed_time:.2f} seconds")
    logger.info(f"  * Stored Coreset Memory: {meta['coreset_size']} vectors (Dim: {meta['feature_dim']})")
    logger.info(f"  * Smoke Test Score:      {res['anomaly_score']:.4f}")
    logger.info(f"  * Heatmap Dimensions:    {res['anomaly_map'].shape}")
    logger.info(f"  * Checkpoint Location:   {saved_path}")
    logger.info("=" * 65)
    print("\n[SUCCESS] PatchCore model training and memory bank serialization verified.")


if __name__ == "__main__":
    main()
