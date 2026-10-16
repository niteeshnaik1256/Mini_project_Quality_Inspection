"""
FastAPI Inference Endpoint for Quality Inspector.
Provides REST API for single-image and video-frame inspection.
Integrates with the multi-role Smart Factory Monitoring system.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events.quality_event import QualityCVEvent
from src.roi.pipeline import QualityInspectionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QualityInspectorAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Quality Inspector API",
    description="Industrial Automotive Visual Quality Inspection API",
    version="1.0.0",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded)
pipeline: Optional[QualityInspectionPipeline] = None


# Pydantic models for request/response
class InspectionRequest(BaseModel):
    camera_id: str = "default"
    product_id: str = "unknown"
    save_evidence: bool = True


class InspectionResponse(BaseModel):
    event_id: str
    camera_id: str
    role: str
    task: str
    timestamp: str
    product_id: str
    anomaly_score: float
    decision: str
    severity: str
    bbox: List[int]
    track_id: Optional[str]
    evidence_path: str
    heatmap_path: str
    metadata: Dict[str, Any]


class PipelineConfig(BaseModel):
    model_checkpoint: str
    decision_calibration_path: Optional[str] = None
    roi_config: Optional[Dict[str, Any]] = None
    enable_temporal: bool = False
    output_dir: str = "results/inspections"


@app.on_event("startup")
async def startup_event():
    """Initialize the quality inspection pipeline on startup."""
    global pipeline

    # Default configuration (can be overridden via /configure endpoint)
    default_config = PipelineConfig(
        model_checkpoint=str(PROJECT_ROOT / "models" / "patchcore" / "patchcore_checkpoint.pt"),
        decision_calibration_path=str(PROJECT_ROOT / "results" / "reports" / "decision_calibration.json"),
        enable_temporal=False,
        output_dir=str(PROJECT_ROOT / "results" / "inspections"),
    )

    try:
        pipeline = QualityInspectionPipeline(
            model_checkpoint=default_config.model_checkpoint,
            decision_calibration_path=default_config.decision_calibration_path,
            roi_config=default_config.roi_config,
            enable_temporal=default_config.enable_temporal,
            output_dir=default_config.output_dir,
        )
        logger.info("Quality Inspection Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # Don't raise - allow manual configuration via /configure


@app.post("/configure", response_model=Dict[str, str])
async def configure_pipeline(config: PipelineConfig):
    """
    Configure or reconfigure the quality inspection pipeline.
    Use this endpoint to change models, calibration, or settings at runtime.
    """
    global pipeline

    try:
        pipeline = QualityInspectionPipeline(
            model_checkpoint=config.model_checkpoint,
            decision_calibration_path=config.decision_calibration_path,
            roi_config=config.roi_config,
            enable_temporal=config.enable_temporal,
            output_dir=config.output_dir,
        )
        logger.info(f"Pipeline reconfigured with model: {config.model_checkpoint}")
        return {"status": "success", "message": "Pipeline configured successfully"}
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@app.post("/inspect/image", response_model=InspectionResponse)
async def inspect_image(
    file: UploadFile = File(...),
    camera_id: str = Form("default"),
    product_id: str = Form("unknown"),
    save_evidence: bool = Form(True),
):
    """
    Inspect a single uploaded image for quality anomalies.

    Args:
        file: Image file to inspect
        camera_id: Camera identifier
        product_id: Product/component identifier
        save_evidence: Whether to save evidence images

    Returns:
        QualityCVEvent with inspection results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Configure via /configure")

    try:
        # Read uploaded file
        contents = await file.read()
        import numpy as np
        import cv2

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Run inspection
        event = pipeline.inspect_image(
            image=image,
            camera_id=camera_id,
            product_id=product_id,
            save_evidence=save_evidence,
        )

        return InspectionResponse(**event.to_dict())

    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")


@app.post("/inspect/frame", response_model=InspectionResponse)
async def inspect_video_frame(
    file: UploadFile = File(...),
    frame_idx: int = Form(0),
    timestamp: float = Form(0.0),
    camera_id: str = Form("default"),
    product_id: str = Form("unknown"),
    save_evidence: bool = Form(True),
):
    """
    Inspect a video frame with temporal validation.

    Args:
        file: Video frame as image file
        frame_idx: Frame number in video
        timestamp: Frame timestamp in seconds
        camera_id: Camera identifier
        product_id: Product identifier
        save_evidence: Whether to save evidence

    Returns:
        QualityCVEvent with inspection results (if event triggered)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Configure via /configure")

    try:
        # Read uploaded file
        contents = await file.read()
        import numpy as np
        import cv2

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")

        # Run inspection with temporal filtering
        _, events = pipeline.inspect_video_frame(
            frame=frame,
            frame_idx=frame_idx,
            timestamp=timestamp,
            camera_id=camera_id,
            product_id=product_id,
            save_evidence=save_evidence,
        )

        # Return first event if any triggered
        if events:
            return InspectionResponse(**events[0].to_dict())
        else:
            # Return empty event if no event triggered
            return InspectionResponse(
                event_id="",
                camera_id=camera_id,
                role="quality",
                task="anomaly_inspection",
                timestamp="",
                product_id=product_id,
                anomaly_score=0.0,
                decision="PASS",
                severity="low",
                bbox=[],
                track_id=None,
                evidence_path="",
                heatmap_path="",
                metadata={"message": "No event triggered (temporal persistence not met)"},
            )

    except Exception as e:
        logger.error(f"Frame inspection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Frame inspection failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "temporal_enabled": pipeline.enable_temporal if pipeline else False,
    }


@app.get("/stats")
async def get_stats():
    """Get inspection statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_inspection_stats()
    return stats


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Quality Inspector API",
        "version": "1.0.0",
        "description": "Industrial Automotive Visual Quality Inspection",
        "endpoints": {
            "POST /configure": "Configure the inspection pipeline",
            "POST /inspect/image": "Inspect a single image",
            "POST /inspect/frame": "Inspect a video frame with temporal validation",
            "GET /health": "Health check",
            "GET /stats": "Get inspection statistics",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
