"""
PaDiM Training Script.
Calibrates per-patch multivariate Gaussians on nominal AutoVI training images
and serializes mean / inverse-covariance statistics to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders
from src.models.padim import PaDiMModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainPaDiM")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaDiM Gaussian Patch Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "padim.yaml"),
        help="Path to PaDiM YAML configuration",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Override number of randomly selected feature dimensions",
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

    if args.n_features is not None:
        config["model"]["n_features"] = args.n_features
    if args.output_dir is not None:
        config["trainer"]["output_dir"] = args.output_dir

    seed = config["trainer"].get("seed", 42)
    set_seed(seed)

    logger.info("=" * 65)
    logger.info("   INDUSTRIAL AOI: PADIM GAUSSIAN PATCH BASELINE")
    logger.info("=" * 65)
    logger.info(f"Target Category:        {config['dataset']['category']}")
    logger.info(f"Backbone:               {config['model']['backbone']}")
    logger.info(f"Feature Layers:         {config['model']['layers']}")
    logger.info(f"Selected Dimensions:    {config['model']['n_features']}")
    logger.info(f"Image Resolution:       {config['dataset']['image_size']}")
    logger.info("-" * 65)

    train_loader, test_loader = build_dataloaders(config)
    logger.info(
        f"Nominal training batches: {len(train_loader)} "
        f"(batch size: {config['dataset']['train_batch_size']})"
    )

    device = "cuda" if torch.cuda.is_available() and config["trainer"]["device"] == "auto" else "cpu"
    logger.info(f"Compute Device:         {device.upper()}")

    model = PaDiMModel(
        backbone_name=config["model"]["backbone"],
        layers=tuple(config["model"]["layers"]),
        n_features=config["model"]["n_features"],
        cov_regularizer=config["model"].get("cov_regularizer", 1e-3),
        feature_selection_seed=config["model"].get("feature_selection_seed", 42),
        device=device,
    )

    start_time = time.time()
    model.fit(train_loader)
    elapsed_time = time.time() - start_time

    out_dir = Path(config["trainer"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    ckpt_name = config["trainer"].get("checkpoint_name", "padim_checkpoint.pt")
    ckpt_path = out_dir / ckpt_name

    h, w = model.spatial_shape or (0, 0)
    meta = {
        "category": config["dataset"]["category"],
        "training_samples": len(train_loader.dataset),
        "fit_time_seconds": round(elapsed_time, 2),
        "spatial_grid": [h, w],
        "n_features": model.feature_indices.shape[0] if model.feature_indices is not None else 0,
        "cov_regularizer": config["model"].get("cov_regularizer", 1e-3),
    }
    saved_path = model.save_checkpoint(ckpt_path, extra_metadata=meta)

    logger.info("Executing smoke-test inference on single test sample...")
    test_sample = next(iter(test_loader))
    sample_img = test_sample["image"][0:1]
    blur_sigma = config.get("evaluation", {}).get("gaussian_blur_sigma", 4.0)
    res = model.predict_step(sample_img, blur_sigma=blur_sigma)

    logger.info("-" * 65)
    logger.info("TRAINING AUDIT & BENCHMARK SUMMARY:")
    logger.info(f"  * Elapsed Calibration Time: {elapsed_time:.2f} seconds")
    logger.info(f"  * Patch Grid:               {h} x {w}")
    logger.info(f"  * Feature Subspace Dim:     {meta['n_features']}")
    logger.info(f"  * Smoke Test Score:         {res['anomaly_score']:.4f}")
    logger.info(f"  * Heatmap Dimensions:       {res['anomaly_map'].shape}")
    logger.info(f"  * Checkpoint Location:      {saved_path}")
    logger.info("=" * 65)
    print("\n[SUCCESS] PaDiM Gaussian patch model calibration verified.")


if __name__ == "__main__":
    main()
