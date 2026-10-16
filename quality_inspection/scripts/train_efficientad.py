"""
EfficientAD Training Script.
Distills teacher feature representations into a lightweight student network
trained exclusively on nominal automotive components.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders
from src.models.efficientad import EfficientADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainEfficientAD")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train EfficientAD Student-Teacher Model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "efficientad.yaml"),
        help="Path to EfficientAD YAML configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    logger.info("=" * 65)
    logger.info("   INDUSTRIAL AOI: EFFICIENTAD DISTILLATION TRAINING")
    logger.info("=" * 65)
    logger.info(f"Target Category:        {config['dataset']['category']}")
    logger.info(f"Teacher Backbone:       {config['model']['teacher_backbone']}")
    logger.info(f"Feature Dimension:      {config['model']['feature_dim']}")
    logger.info(f"Training Epochs:        {config['training']['epochs']}")
    logger.info(f"Learning Rate:          {config['training']['learning_rate']}")
    logger.info("-" * 65)

    # 1. Build DataLoaders
    train_loader, test_loader = build_dataloaders(config)
    device = "cuda" if torch.cuda.is_available() and config["training"]["device"] == "auto" else "cpu"
    logger.info(f"Compute Device:         {device.upper()}")

    # 2. Instantiate Model
    model = EfficientADModel(
        teacher_backbone=config["model"]["teacher_backbone"],
        feature_layer=config["model"]["feature_layer"],
        feature_dim=config["model"]["feature_dim"],
        device=device,
    )

    # 3. Calibrate Normalization Stats
    model.compute_normalization_stats(train_loader)

    # 4. Train Student Network via Distillation Loss
    optimizer = optim.Adam(
        model.student.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )
    criterion = nn.MSELoss()

    epochs = config["training"]["epochs"]
    start_time = time.time()
    model.student.train()

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)

            with torch.no_grad():
                t_feat = model.extract_teacher_features(images)
                t_norm = model.normalize_teacher_features(t_feat)

            s_feat = model.student(images)
            loss = criterion(s_feat, t_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] — Distillation Loss: {mean_loss:.6f}")

    elapsed_time = time.time() - start_time

    # 5. Save Checkpoint
    out_dir = Path(config["training"]["output_dir"])
    ckpt_name = config["training"].get("checkpoint_name", "efficientad_checkpoint.pt")
    ckpt_path = out_dir / ckpt_name

    meta = {
        "category": config["dataset"]["category"],
        "training_epochs": epochs,
        "final_loss": round(mean_loss, 6),
        "fit_time_seconds": round(elapsed_time, 2),
    }
    saved_path = model.save_checkpoint(ckpt_path, extra_metadata=meta)

    # 6. Smoke Test on 1 Test Sample
    model.eval()
    test_sample = next(iter(test_loader))
    sample_img = test_sample["image"][0:1]
    res = model.predict_step(sample_img)

    logger.info("-" * 65)
    logger.info("TRAINING AUDIT & BENCHMARK SUMMARY:")
    logger.info(f"  * Distillation Time:     {elapsed_time:.2f} seconds")
    logger.info(f"  * Final Distill Loss:    {mean_loss:.6f}")
    logger.info(f"  * Smoke Test Score:      {res['anomaly_score']:.4f}")
    logger.info(f"  * Heatmap Dimensions:    {res['anomaly_map'].shape}")
    logger.info(f"  * Checkpoint Location:   {saved_path}")
    logger.info("=" * 65)
    print("\n[SUCCESS] EfficientAD student-teacher model training verified.")


if __name__ == "__main__":
    main()
