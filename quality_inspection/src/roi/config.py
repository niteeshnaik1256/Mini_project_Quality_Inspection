"""
ROI configuration loading for manual (and optional automatic) component cropping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import yaml

CoordMode = Literal["absolute", "normalized"]


@dataclass
class ROIRectangle:
    """Axis-aligned region of interest in image coordinates."""

    x: int
    y: int
    width: int
    height: int
    coord_mode: CoordMode = "absolute"

    def resolve(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Return integer crop box (x1, y1, x2, y2) clamped to image bounds.
        """
        if self.coord_mode == "normalized":
            x1 = int(round(self.x * image_width))
            y1 = int(round(self.y * image_height))
            x2 = int(round((self.x + self.width) * image_width))
            y2 = int(round((self.y + self.height) * image_height))
        else:
            x1, y1 = self.x, self.y
            x2, y2 = self.x + self.width, self.y + self.height

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))
        return x1, y1, x2, y2


@dataclass
class ROIConfig:
    """Quality inspection ROI settings."""

    mode: Literal["manual", "yolo"] = "manual"
    manual: Optional[ROIRectangle] = None
    yolo_weights: Optional[str] = None
    yolo_class_filter: Optional[int] = None
    product_id_default: str = "unknown_part"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROIConfig":
        roi_block = data.get("roi", {})
        mode = roi_block.get("mode", "manual")
        manual_cfg = roi_block.get("manual", {})
        manual = ROIRectangle(
            x=int(manual_cfg.get("x", 0)),
            y=int(manual_cfg.get("y", 0)),
            width=int(manual_cfg.get("width", 256)),
            height=int(manual_cfg.get("height", 256)),
            coord_mode=manual_cfg.get("coord_mode", "absolute"),
        )
        yolo = roi_block.get("yolo", {})
        return cls(
            mode=mode,
            manual=manual,
            yolo_weights=yolo.get("weights"),
            yolo_class_filter=yolo.get("class_filter"),
            product_id_default=data.get("product_id_default", "unknown_part"),
        )


def load_roi_config(path: str | Path) -> ROIConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"ROI config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ROIConfig.from_dict(data)
