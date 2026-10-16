"""
PatchCore Industrial Anomaly Detection Model.
Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
Implements multi-scale patch embeddings, greedy minimax coreset memory bank subsampling,
and nearest-neighbor anomaly scoring with 2D localization heatmaps.
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
logger = logging.getLogger("PatchCore")


class PatchCoreModel(nn.Module):
    """
    Industrial PatchCore Anomaly Detection Architecture.
    Extracts mid-level CNN activations, builds a coreset memory bank of nominal patch embeddings,
    and performs nearest-neighbor scoring for defect detection and pixel-level localization.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        patch_size: int = 3,
        patch_stride: int = 1,
        coreset_sampling_ratio: float = 0.10,
        num_neighbors: int = 9,
        dimension_reduction: int = 128,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.dimension_reduction = dimension_reduction

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize CNN Backbone
        self.backbone = self._load_backbone(backbone_name)
        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Register Hooks for Intermediate Feature Maps
        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # 3. Patch Pooling Operator
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=self.patch_size // 2,
        )

        # 4. Memory Bank and Projection Matrix
        self.memory_bank: Optional[torch.Tensor] = None
        self.random_projection: Optional[torch.Tensor] = None
        self.embedding_dimension: Optional[int] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None

    def _load_backbone(self, name: str) -> nn.Module:
        """Load pretrained CNN backbone with graceful offline fallback."""
        if name.lower() == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                logger.info("Loaded pretrained ResNet18 weights.")
            except Exception as e:
                logger.warning(f"Could not download remote weights ({e}); using initialized ResNet18.")
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
        """Attach forward hooks to target intermediate feature layers."""
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._hook_fn(layer_name))

    def _hook_fn(self, layer_name: str) -> Any:
        def hook(_, __, output: torch.Tensor):
            self.features[layer_name] = output
        return hook

    def extract_patch_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract and aggregate locally-aware patch embeddings from intermediate feature maps.
        Returns:
            embeddings: (B, H_ref * W_ref, C_total)
            spatial_shape: (H_ref, W_ref)
        """
        x = x.to(self.device)
        self.features.clear()
        with torch.no_grad():
            _ = self.backbone(x)

        extracted_maps: List[torch.Tensor] = []
        ref_h, ref_w = 0, 0

        for i, layer_name in enumerate(self.layers):
            feat = self.features[layer_name]
            # Locally-aware neighbourhood aggregation
            pooled = self.avg_pool(feat)
            if i == 0:
                ref_h, ref_w = pooled.shape[-2], pooled.shape[-1]
                extracted_maps.append(pooled)
            else:
                # Bilinear resize to align with reference spatial resolution
                resized = F.interpolate(
                    pooled,
                    size=(ref_h, ref_w),
                    mode="bilinear",
                    align_corners=False,
                )
                extracted_maps.append(resized)

        # Concatenate along channel dimension
        concatenated = torch.cat(extracted_maps, dim=1)  # (B, C_total, H_ref, W_ref)
        b, c, h, w = concatenated.shape
        # Reshape to (B, H*W, C)
        embeddings = concatenated.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return embeddings, (h, w)

    def _minimax_coreset_sampling(
        self,
        embeddings: torch.Tensor,
        sampling_ratio: float,
    ) -> torch.Tensor:
        """
        Greedy K-Center Minimax Coreset Subsampling.
        Selects a representative memory bank subset minimizing maximum coverage distance.
        Uses random projection for computationally efficient distance calculations.
        """
        n_samples, feat_dim = embeddings.shape
        target_size = max(1, int(n_samples * sampling_ratio))
        logger.info(f"Subsampling memory bank: {n_samples} patches -> {target_size} coreset patches (ratio: {sampling_ratio:.2%})")

        if target_size >= n_samples:
            return embeddings

        # Random Projection for fast L2 distance evaluation
        proj_dim = min(self.dimension_reduction, feat_dim)
        torch.manual_seed(42)
        proj_matrix = torch.randn(feat_dim, proj_dim, device=self.device)
        proj_matrix = F.normalize(proj_matrix, p=2, dim=0)

        # Project features
        projected = torch.matmul(embeddings, proj_matrix)  # (N, proj_dim)

        # Initialize coreset with the point furthest from the centroid
        centroid = torch.mean(projected, dim=0, keepdim=True)
        dist_to_centroid = torch.norm(projected - centroid, dim=1)
        first_idx = torch.argmax(dist_to_centroid).item()

        selected_indices = [first_idx]
        min_distances = torch.norm(projected - projected[first_idx : first_idx + 1], dim=1)

        # Iterative greedy minimax selection
        for step in range(1, target_size):
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

            # Update distance vector
            new_distances = torch.norm(projected - projected[next_idx : next_idx + 1], dim=1)
            min_distances = torch.minimum(min_distances, new_distances)

            if (step + 1) % max(1, target_size // 5) == 0 or step == target_size - 1:
                logger.info(f"Coreset selection progress: {step + 1}/{target_size} patches selected.")

        coreset = embeddings[selected_indices]
        return coreset

    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        """
        Construct memory bank from nominal training samples.
        Extracts patch embeddings across all defect-free training images and applies coreset reduction.
        """
        logger.info("Extracting patch embeddings from nominal training partition...")
        all_embeddings: List[torch.Tensor] = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            emb, spatial_shape = self.extract_patch_embeddings(images)
            self.spatial_shape = spatial_shape
            b, num_patches, c = emb.shape
            all_embeddings.append(emb.reshape(-1, c))

        full_memory_bank = torch.cat(all_embeddings, dim=0)
        self.embedding_dimension = full_memory_bank.shape[-1]
        logger.info(f"Aggregated raw patch memory bank: {full_memory_bank.shape[0]} vectors of dimension {self.embedding_dimension}.")

        # Perform coreset subsampling
        self.memory_bank = self._minimax_coreset_sampling(
            full_memory_bank,
            self.coreset_sampling_ratio,
        )
        logger.info(f"PatchCore training complete. Active memory bank size: {self.memory_bank.shape}.")

    def predict_step(
        self,
        image_tensor: torch.Tensor,
        blur_sigma: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Compute anomaly score and pixel-level heatmap for a single test image.
        Args:
            image_tensor: (1, 3, H, W) normalized tensor
        Returns:
            dict containing:
                "anomaly_score": float (image-level anomaly score)
                "anomaly_map": np.ndarray (H, W) smoothed 2D anomaly heatmap
        """
        if self.memory_bank is None:
            raise RuntimeError("Model must be trained (fit) or checkpoint loaded before inference.")

        self.backbone.eval()
        image_tensor = image_tensor.to(self.device)
        _, _, orig_h, orig_w = image_tensor.shape

        # Extract test patch embeddings
        emb, (h_feat, w_feat) = self.extract_patch_embeddings(image_tensor)
        patches = emb.squeeze(0)  # (h_feat * w_feat, D)

        # Nearest-Neighbor Euclidean Distance to Memory Bank
        # ||p - m||^2 = ||p||^2 - 2 p m^T + ||m||^2
        p_norm = torch.sum(patches ** 2, dim=1, keepdim=True)       # (P, 1)
        m_norm = torch.sum(self.memory_bank ** 2, dim=1, keepdim=True).T  # (1, M)
        dists_sq = p_norm + m_norm - 2.0 * torch.matmul(patches, self.memory_bank.T)
        dists = torch.sqrt(torch.clamp(dists_sq, min=1e-8))         # (P, M)

        # Minimum distance per patch
        min_patch_dists, _ = torch.min(dists, dim=1)                 # (P,)
        anomaly_score = torch.max(min_patch_dists).item()

        # Reshape to 2D feature grid
        score_map = min_patch_dists.reshape(1, 1, h_feat, w_feat)

        # Bilinear upsampling to original image resolution
        upsampled_map = F.interpolate(
            score_map,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Gaussian Smoothing to prevent pixel-level checkerboard artifacts
        if blur_sigma > 0:
            smoothed_map = ndimage.gaussian_filter(upsampled_map, sigma=blur_sigma)
        else:
            smoothed_map = upsampled_map

        return {
            "anomaly_score": float(anomaly_score),
            "anomaly_map": smoothed_map,
        }

    def save_checkpoint(self, filepath: str | Path, extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Serialize memory bank and configuration to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_name": "patchcore",
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "patch_size": self.patch_size,
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "embedding_dimension": self.embedding_dimension,
            "spatial_shape": self.spatial_shape,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, out_path)
        logger.info(f"PatchCore checkpoint saved to: {out_path}")
        return out_path

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Restore memory bank and configuration from checkpoint."""
        ckpt_path = Path(filepath)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.memory_bank = checkpoint["memory_bank"].to(self.device)
        self.embedding_dimension = checkpoint["embedding_dimension"]
        self.spatial_shape = checkpoint["spatial_shape"]
        logger.info(f"Restored PatchCore memory bank of size {self.memory_bank.shape} from {ckpt_path}")
