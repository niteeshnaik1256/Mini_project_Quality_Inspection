"""
Industrial Inference Engine for Automotive Visual Inspection.
Executes low-latency PatchCore inference on camera frames, image arrays, or file paths.
Produces image-level anomaly scores, pixel-level localization heatmaps, and latency benchmarks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.patchcore import PatchCoreModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PatchCoreInferencer")


class PatchCoreInferencer:
    """
    Production-oriented inference worker for automotive optical inspection.
    Provides fast, deterministic evaluation of component images.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        gaussian_blur_sigma: float = 4.0,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {self.checkpoint_path}")

        self.image_size = image_size
        self.gaussian_blur_sigma = gaussian_blur_sigma

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing Inferencer on device: {self.device.upper()}")
        self._load_model()

        # Deterministic preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self) -> None:
        """Load checkpoint metadata and initialize appropriate model architecture."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model_type = ckpt.get("model_name", "patchcore").lower()

        if model_type == "efficientad":
            from src.models.efficientad import EfficientADModel
            teacher_backbone = ckpt.get("teacher_backbone", "resnet18")
            feature_layer = ckpt.get("feature_layer", "layer2")
            feature_dim = ckpt.get("feature_dim", 128)

            self.model = EfficientADModel(
                teacher_backbone=teacher_backbone,
                feature_layer=feature_layer,
                feature_dim=feature_dim,
                device=self.device,
            )
            self.model.student.load_state_dict(ckpt["student_state_dict"])
            self.model.teacher_mean.copy_(ckpt["teacher_mean"].to(self.device))
            self.model.teacher_std.copy_(ckpt["teacher_std"].to(self.device))
            self.model.eval()
            logger.info("Loaded EfficientAD Student-Teacher model successfully.")

        elif model_type == "padim":
            from src.models.padim import PaDiMModel
            backbone_name = ckpt.get("backbone_name", "resnet18")
            layers = tuple(ckpt.get("layers", ("layer2", "layer3")))
            n_features = ckpt.get("n_features", 100)

            self.model = PaDiMModel(
                backbone_name=backbone_name,
                layers=layers,
                n_features=n_features,
                cov_regularizer=ckpt.get("cov_regularizer", 1e-3),
                feature_selection_seed=ckpt.get("feature_selection_seed", 42),
                device=self.device,
            )
            self.model.feature_indices = ckpt["feature_indices"].to(self.device)
            self.model.total_feature_dim = ckpt["total_feature_dim"]
            self.model.spatial_shape = ckpt.get("spatial_shape")
            self.model.mean = ckpt["mean"].to(self.device)
            self.model.inv_covariance = ckpt["inv_covariance"].to(self.device)
            self.model.eval()
            logger.info(
                f"Loaded PaDiM '{backbone_name}' with grid {self.model.spatial_shape} "
                f"and {self.model.feature_indices.shape[0]} feature dimensions."
            )

        else:  # Default: patchcore
            backbone_name = ckpt.get("backbone_name", "resnet18")
            layers = tuple(ckpt.get("layers", ("layer2", "layer3")))
            patch_size = ckpt.get("patch_size", 3)

            self.model = PatchCoreModel(
                backbone_name=backbone_name,
                layers=layers,
                patch_size=patch_size,
                device=self.device,
            )
            self.model.memory_bank = ckpt["memory_bank"].to(self.device)
            self.model.embedding_dimension = ckpt["embedding_dimension"]
            self.model.spatial_shape = ckpt.get("spatial_shape")
            self.model.eval()
            logger.info(
                f"Loaded PatchCore '{backbone_name}' with memory bank of size {self.model.memory_bank.shape}."
            )

    def _preprocess_input(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Normalize input into a 4D torch Tensor (1, 3, H, W).
        Returns tensor and original image dimensions (orig_h, orig_w).
        """
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            pil_img = Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assume RGB or BGR from OpenCV
            if image_input.ndim == 2:  # Grayscale
                rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            elif image_input.shape[2] == 4:
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")
            pil_img = Image.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        orig_w, orig_h = pil_img.size
        tensor = self.transform(pil_img).unsqueeze(0)  # (1, 3, H, W)
        return tensor, (orig_h, orig_w)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Execute anomaly inspection for a single automotive image.
        Returns:
            dict containing:
                "anomaly_score": float
                "anomaly_map": np.ndarray (H, W) raw anomaly heatmap
                "latency_ms": float
                "fps": float
                "original_shape": Tuple[int, int]
                "device": str
        """
        t0 = time.perf_counter()
        tensor, (orig_h, orig_w) = self._preprocess_input(image_input)

        with torch.no_grad():
            res = self.model.predict_step(
                tensor,
                blur_sigma=self.gaussian_blur_sigma,
            )

        latency_sec = time.perf_counter() - t0
        latency_ms = latency_sec * 1000.0
        fps = 1.0 / max(latency_sec, 1e-6)

        # Ensure anomaly map matches original input aspect ratio
        heatmap = res["anomaly_map"]
        if heatmap.shape != (orig_h, orig_w):
            heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return {
            "anomaly_score": float(res["anomaly_score"]),
            "anomaly_map": heatmap,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
            "original_shape": (orig_h, orig_w),
            "device": self.device,
        }

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run sequential low-latency inference over a batch of images."""
        return [self.predict(img) for img in image_inputs]
