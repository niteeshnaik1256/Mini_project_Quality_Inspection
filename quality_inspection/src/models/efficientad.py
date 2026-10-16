"""
EfficientAD: Low-Latency Student-Teacher Distillation for Industrial Anomaly Detection.
Reference: Batzner et al., "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies", WACV 2024.
Replaces slow nearest-neighbor memory banks with feed-forward student-teacher feature discrepancy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EfficientAD")


class StudentNetwork(nn.Module):
    """
    Lightweight feed-forward convolutional student network.
    Maps input RGB images (3, H, W) to match teacher feature map dimensions (C, H/8, W/8).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # H/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # H/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # H/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientADModel(nn.Module):
    """
    Student-Teacher Anomaly Detection Architecture.
    Teacher: Pretrained frozen CNN backbone.
    Student: Trained on nominal images to predict normalized teacher representations.
    Discrepancy between student and teacher directly yields the anomaly map.
    """

    def __init__(
        self,
        teacher_backbone: str = "resnet18",
        feature_layer: str = "layer2",
        feature_dim: int = 128,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.teacher_backbone_name = teacher_backbone
        self.feature_layer = feature_layer
        self.feature_dim = feature_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize Teacher Network (Frozen)
        self.teacher = self._load_teacher(teacher_backbone)
        self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher_features: Dict[str, torch.Tensor] = {}
        getattr(self.teacher, self.feature_layer).register_forward_hook(self._hook_fn())

        # 2. Initialize Student Network (Trainable)
        self.student = StudentNetwork(in_channels=3, out_channels=self.feature_dim)
        self.student.to(self.device)

        # 3. Running Feature Normalization Parameters
        self.register_buffer("teacher_mean", torch.zeros(1, self.feature_dim, 1, 1, device=self.device))
        self.register_buffer("teacher_std", torch.ones(1, self.feature_dim, 1, 1, device=self.device))

    def _load_teacher(self, name: str) -> nn.Module:
        """Load pretrained teacher network with offline fallback."""
        if name.lower() == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                logger.info("Loaded pretrained ResNet18 teacher weights.")
            except Exception:
                model = models.resnet18(weights=None)
            return model
        else:
            raise ValueError(f"Unsupported teacher backbone: {name}")

    def _hook_fn(self) -> Any:
        def hook(_, __, output: torch.Tensor):
            self.teacher_features["feat"] = output
        return hook

    def extract_teacher_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from frozen teacher."""
        self.teacher_features.clear()
        with torch.no_grad():
            _ = self.teacher(x)
        feat = self.teacher_features["feat"]
        return feat

    def compute_normalization_stats(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Calculate mean and standard deviation of teacher features over nominal training set."""
        logger.info("Calibrating teacher feature normalization statistics...")
        all_feats: List[torch.Tensor] = []
        for batch in train_loader:
            imgs = batch["image"].to(self.device)
            feat = self.extract_teacher_features(imgs)
            all_feats.append(feat.detach())

        stacked = torch.cat(all_feats, dim=0)  # (N, C, H, W)
        mean = stacked.mean(dim=(0, 2, 3), keepdim=True)
        std = stacked.std(dim=(0, 2, 3), keepdim=True) + 1e-4

        self.teacher_mean.copy_(mean)
        self.teacher_std.copy_(std)
        logger.info(f"Teacher normalization calibrated: mean shape {self.teacher_mean.shape}, std shape {self.teacher_std.shape}.")

    def normalize_teacher_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Standardize teacher features to zero-mean unit-variance."""
        return (feat - self.teacher_mean) / self.teacher_std

    def compute_anomaly_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pixel-wise squared feature discrepancy between student and teacher.
        Returns:
            anomaly_map: (B, 1, H, W) upsampled distance field
            patch_distance: (B, 1, H_feat, W_feat) raw feature distance
        """
        b, _, orig_h, orig_w = x.shape
        t_feat = self.extract_teacher_features(x)
        t_norm = self.normalize_teacher_features(t_feat)
        s_feat = self.student(x)

        # Squared difference averaged across channels
        dist = torch.mean((s_feat - t_norm) ** 2, dim=1, keepdim=True)  # (B, 1, H/8, W/8)

        # Bilinear upsample to full input image resolution
        upsampled = F.interpolate(
            dist,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled, dist

    def predict_step(
        self,
        image_tensor: torch.Tensor,
        blur_sigma: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Feed-forward anomaly scoring for a single image tensor (1, 3, H, W).
        Achieves ultra-low latency with zero nearest-neighbor search.
        """
        self.eval()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            up_map, _ = self.compute_anomaly_map(image_tensor)

        raw_map = up_map.squeeze().cpu().numpy()
        anomaly_score = float(np.max(raw_map))

        # Gaussian smoothing
        if blur_sigma > 0:
            smoothed = ndimage.gaussian_filter(raw_map, sigma=blur_sigma)
        else:
            smoothed = raw_map

        return {
            "anomaly_score": float(np.max(smoothed)),
            "anomaly_map": smoothed,
        }

    def save_checkpoint(self, filepath: str | Path, extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Serialize student weights and teacher normalization statistics."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_name": "efficientad",
            "teacher_backbone": self.teacher_backbone_name,
            "feature_layer": self.feature_layer,
            "feature_dim": self.feature_dim,
            "student_state_dict": self.student.state_dict(),
            "teacher_mean": self.teacher_mean.cpu(),
            "teacher_std": self.teacher_std.cpu(),
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, out_path)
        logger.info(f"EfficientAD checkpoint saved to: {out_path}")
        return out_path

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Restore student weights and normalization statistics."""
        ckpt_path = Path(filepath)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.teacher_mean.copy_(checkpoint["teacher_mean"].to(self.device))
        self.teacher_std.copy_(checkpoint["teacher_std"].to(self.device))
        logger.info(f"Restored EfficientAD student weights from {ckpt_path}")
