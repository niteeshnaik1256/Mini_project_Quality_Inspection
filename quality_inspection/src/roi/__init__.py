"""Region-of-interest extraction for component-level inspection."""

from .config import ROIConfig, ROIRectangle, load_roi_config
from .pipeline import ROIInspectionPipeline, ROIInspectionResult

__all__ = [
    "ROIConfig",
    "ROIRectangle",
    "load_roi_config",
    "ROIInspectionPipeline",
    "ROIInspectionResult",
]
