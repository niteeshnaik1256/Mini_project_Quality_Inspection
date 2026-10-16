"""
Industrial PyTorch Dataset and DataLoader for Automotive Anomaly Detection.
Complies with MVTec / AutoVI one-class learning format.
Guarantees deterministic data loading, paired ground-truth mask handling, and zero leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("IndustrialDataset")


class AutomotiveAnomalyDataset(Dataset):
    """
    Industrial Visual Quality Inspection Dataset for Automotive Components.
    Supports Train (strictly nominal) and Test (nominal + diverse defect archetypes).
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.split = split.lower()
        self.image_size = image_size
        self.is_training = is_training

        if self.split not in {"train", "test", "val"}:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

        # Deterministic transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_samples(self) -> None:
        """Parse directory structure and register image-mask pairs."""
        if self.split == "train":
            train_good_dir = self.root_dir / "train" / "good"
            if not train_good_dir.exists():
                raise FileNotFoundError(f"Required training good directory missing: {train_good_dir}")
            for img_path in sorted(train_good_dir.glob("*")):
                if img_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": None,
                        "label": 0,  # 0: Nominal
                        "defect_type": "good",
                    })
            logger.info(f"Loaded {len(self.samples)} nominal samples for '{self.split}' split.")

        else:  # test or val
            test_dir = self.root_dir / "test"
            gt_dir = self.root_dir / "ground_truth"

            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory missing: {test_dir}")

            # 1. Unseen Nominal Test Samples
            test_good_dir = test_dir / "good"
            if test_good_dir.exists():
                for img_path in sorted(test_good_dir.glob("*")):
                    if img_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        self.samples.append({
                            "image_path": str(img_path),
                            "mask_path": None,
                            "label": 0,  # 0: Nominal
                            "defect_type": "good",
                        })

            # 2. Defect Samples
            for defect_dir in sorted(test_dir.iterdir()):
                if defect_dir.is_dir() and defect_dir.name != "good":
                    d_type = defect_dir.name
                    for img_path in sorted(defect_dir.glob("*")):
                        if img_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                            # Find matching ground truth mask
                            expected_mask_candidates = [
                                gt_dir / d_type / f"{img_path.stem}_mask{img_path.suffix}",
                                gt_dir / d_type / f"{img_path.stem}{img_path.suffix}",
                                gt_dir / d_type / f"{img_path.stem}_mask.png",
                                gt_dir / d_type / f"{img_path.stem}.png",
                            ]
                            mask_file = next((p for p in expected_mask_candidates if p.exists()), None)
                            self.samples.append({
                                "image_path": str(img_path),
                                "mask_path": str(mask_file) if mask_file else None,
                                "label": 1,  # 1: Anomalous
                                "defect_type": d_type,
                            })

            logger.info(f"Loaded {len(self.samples)} samples for '{self.split}' split ({sum(s['label'] == 1 for s in self.samples)} anomalous).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item["image_path"]

        # Load RGB image
        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
            img_tensor = self.transform(img)

        # Load or generate binary ground truth mask
        if item["mask_path"] is not None and Path(item["mask_path"]).exists():
            with Image.open(item["mask_path"]) as raw_mask:
                mask = raw_mask.convert("L")
                mask_tensor = self.mask_transform(mask)
                # Binarize: any value > 0 becomes 1.0
                mask_tensor = (mask_tensor > 0.1).float()
        else:
            # Nominal image -> zero mask
            mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "defect_type": item["defect_type"],
            "image_path": image_path,
        }


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Factory creating PyTorch DataLoaders from configuration dictionary."""
    ds_cfg = config["dataset"]
    root_dir = ds_cfg["root_dir"]
    img_size = tuple(ds_cfg.get("image_size", [256, 256]))
    train_bs = ds_cfg.get("train_batch_size", 8)
    eval_bs = ds_cfg.get("eval_batch_size", 8)
    workers = ds_cfg.get("num_workers", 0)
    seed = ds_cfg.get("seed", 42)

    train_dataset = AutomotiveAnomalyDataset(
        root_dir=root_dir,
        split="train",
        image_size=img_size,
        is_training=True,
    )

    test_dataset = AutomotiveAnomalyDataset(
        root_dir=root_dir,
        split="test",
        image_size=img_size,
        is_training=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=workers,
        generator=g,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )

    return train_loader, test_loader
