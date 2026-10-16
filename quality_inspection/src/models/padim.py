"""
PaDiM: Patch Distribution Modeling for Industrial Anomaly Detection.
Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
Detection and Localization", ICPR 2021.

Fits a multivariate Gaussian over randomly selected CNN feature dimensions at each
spatial patch location on nominal training images. Test-time Mahalanobis distance
yields pixel-level anomaly maps and image-level scores.
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
logger = logging.getLogger("PaDiM")


class PaDiMModel(nn.Module):
    """
    PaDiM anomaly detector with pretrained CNN feature extraction,
    random per-channel subsampling, and location-wise Gaussian modeling.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        n_features: int = 100,
        cov_regularizer: float = 1e-3,
        feature_selection_seed: int = 42,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.n_features = n_features
        self.cov_regularizer = cov_regularizer
        self.feature_selection_seed = feature_selection_seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self._load_backbone(backbone_name)
        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        self.feature_indices: Optional[torch.Tensor] = None
        self.total_feature_dim: Optional[int] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None
        self.mean: Optional[torch.Tensor] = None  # (H, W, D)
        self.inv_covariance: Optional[torch.Tensor] = None  # (H, W, D, D)

    def _load_backbone(self, name: str) -> nn.Module:
        if name.lower() == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                logger.info("Loaded pretrained ResNet18 backbone for PaDiM.")
            except Exception as exc:
                logger.warning(f"Could not load remote weights ({exc}); using initialized ResNet18.")
                model = models.resnet18(weights=None)
            return model
        elif name.lower() == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(weights=None)
            return model
        else:
            raise ValueError(f"Unsupported backbone: {name}. Choose 'resnet18' or 'resnet50'.")

    def _register_hooks(self) -> None:
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._hook_fn(layer_name))

    def _hook_fn(self, layer_name: str) -> Any:
        def hook(_, __, output: torch.Tensor):
            self.features[layer_name] = output

        return hook

    def extract_embedding_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Multi-scale feature concatenation aligned to the first layer's spatial grid.
        Returns:
            embedding_map: (B, C_total, H, W)
            spatial_shape: (H, W)
        """
        x = x.to(self.device)
        self.features.clear()
        with torch.no_grad():
            _ = self.backbone(x)

        maps: List[torch.Tensor] = []
        ref_h, ref_w = 0, 0
        for i, layer_name in enumerate(self.layers):
            feat = self.features[layer_name]
            if i == 0:
                ref_h, ref_w = feat.shape[-2], feat.shape[-1]
                maps.append(feat)
            else:
                resized = F.interpolate(
                    feat,
                    size=(ref_h, ref_w),
                    mode="bilinear",
                    align_corners=False,
                )
                maps.append(resized)

        concatenated = torch.cat(maps, dim=1)
        return concatenated, (ref_h, ref_w)

    def _select_feature_indices(self, total_channels: int) -> torch.Tensor:
        """Randomly subsample feature dimensions (fixed seed for reproducibility)."""
        if total_channels <= self.n_features:
            indices = torch.arange(total_channels)
            logger.info(
                f"Using all {total_channels} feature channels (requested n_features={self.n_features})."
            )
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.feature_selection_seed)
            indices = torch.randperm(total_channels, generator=gen)[: self.n_features]
            indices, _ = torch.sort(indices)
            logger.info(
                f"Randomly selected {self.n_features}/{total_channels} PaDiM feature dimensions."
            )
        return indices.to(self.device)

    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        """
        Estimate per-location mean and inverse covariance from nominal training images.
        """
        logger.info("Extracting PaDiM training embeddings from nominal partition...")
        embedding_batches: List[torch.Tensor] = []

        for batch in train_loader:
            images = batch["image"].to(self.device)
            feat_map, spatial_shape = self.extract_embedding_map(images)
            self.spatial_shape = spatial_shape
            if self.feature_indices is None:
                self.total_feature_dim = feat_map.shape[1]
                self.feature_indices = self._select_feature_indices(self.total_feature_dim)

            selected = feat_map[:, self.feature_indices, :, :].detach().cpu()
            embedding_batches.append(selected)

        # (N_images, D, H, W)
        stacked = torch.cat(embedding_batches, dim=0)
        n_images, d, h, w = stacked.shape
        logger.info(
            f"PaDiM calibration tensor: {n_images} images, {d} dims, grid {h}x{w} "
            f"({h * w} patch locations)."
        )

        # (H, W, N, D)
        per_location = stacked.permute(2, 3, 0, 1)
        mean = torch.zeros(h, w, d, dtype=torch.float32)
        inv_cov = torch.zeros(h, w, d, d, dtype=torch.float32)

        eye = torch.eye(d, dtype=torch.float32)
        reg = self.cov_regularizer

        for i in range(h):
            for j in range(w):
                samples = per_location[i, j]  # (N, D)
                mu = samples.mean(dim=0)
                centered = samples - mu.unsqueeze(0)
                if n_images > 1:
                    cov = (centered.T @ centered) / max(n_images - 1, 1)
                else:
                    cov = torch.zeros(d, d)
                cov = cov + reg * eye
                try:
                    inv = torch.linalg.inv(cov)
                except RuntimeError:
                    inv = torch.linalg.pinv(cov)
                mean[i, j] = mu
                inv_cov[i, j] = inv

            if (i + 1) % max(1, h // 4) == 0 or i == h - 1:
                logger.info(f"Gaussian fit progress: row {i + 1}/{h}")

        self.mean = mean.to(self.device)
        self.inv_covariance = inv_cov.to(self.device)
        logger.info("PaDiM training complete: per-patch Gaussian parameters stored.")

    def _mahalanobis_map(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for each spatial patch. Returns (H, W)."""
        if self.mean is None or self.inv_covariance is None:
            raise RuntimeError("Model must be fit or checkpoint loaded before inference.")

        selected = feat_map[:, self.feature_indices, :, :]  # (1, D, H, W)
        _, d, h, w = selected.shape
        patches = selected.squeeze(0).permute(1, 2, 0)  # (H, W, D)

        diff = patches - self.mean
        # (H, W, D) @ (H, W, D, D) -> (H, W, D)
        maha = torch.einsum("hwi,hwij,hwj->hw", diff, self.inv_covariance, diff)
        return torch.clamp(maha, min=0.0)

    def predict_step(
        self,
        image_tensor: torch.Tensor,
        blur_sigma: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Compute image-level score (max patch Mahalanobis distance) and localization heatmap.
        """
        self.backbone.eval()
        image_tensor = image_tensor.to(self.device)
        _, _, orig_h, orig_w = image_tensor.shape

        feat_map, _ = self.extract_embedding_map(image_tensor)
        score_grid = self._mahalanobis_map(feat_map)
        anomaly_score = float(torch.max(score_grid).item())

        score_map = score_grid.unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            score_map,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        if blur_sigma > 0:
            smoothed = ndimage.gaussian_filter(upsampled, sigma=blur_sigma)
        else:
            smoothed = upsampled

        return {
            "anomaly_score": float(np.max(smoothed)),
            "anomaly_map": smoothed,
        }

    def save_checkpoint(self, filepath: str | Path, extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_name": "padim",
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "n_features": self.n_features,
            "cov_regularizer": self.cov_regularizer,
            "feature_selection_seed": self.feature_selection_seed,
            "feature_indices": self.feature_indices.cpu() if self.feature_indices is not None else None,
            "total_feature_dim": self.total_feature_dim,
            "spatial_shape": self.spatial_shape,
            "mean": self.mean.cpu() if self.mean is not None else None,
            "inv_covariance": self.inv_covariance.cpu() if self.inv_covariance is not None else None,
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, out_path)
        logger.info(f"PaDiM checkpoint saved to: {out_path}")
        return out_path

    def load_checkpoint(self, filepath: str | Path) -> None:
        ckpt_path = Path(filepath)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.feature_indices = checkpoint["feature_indices"].to(self.device)
        self.total_feature_dim = checkpoint["total_feature_dim"]
        self.spatial_shape = checkpoint.get("spatial_shape")
        self.mean = checkpoint["mean"].to(self.device)
        self.inv_covariance = checkpoint["inv_covariance"].to(self.device)
        logger.info(
            f"Restored PaDiM Gaussian parameters for grid {self.spatial_shape} "
            f"with {self.feature_indices.shape[0]} feature dimensions."
        )
