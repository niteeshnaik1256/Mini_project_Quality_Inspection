# Quality Inspector Module

Industrial-grade visual quality inspection for automotive components in the Smart Factory Monitoring system.

## Overview

The Quality Inspector module provides automated optical inspection (AOI) for automotive manufacturing lines. It detects visual anomalies in components using state-of-the-art anomaly detection models and generates standardized quality events compatible with the multi-role Smart Factory system.

## Architecture

```
Camera/Image
    ↓
ROI Extraction (Manual/YOLO)
    ↓
Anomaly Detection (PatchCore/EfficientAD/PaDiM)
    ↓
Decision Engine (PASS/REVIEW/FAIL)
    ↓
Temporal Validation (for video streams)
    ↓
Event Generation (Standardized CVEvent)
    ↓
Evidence Saving (images, heatmaps)
```

## Features

- **Multi-Model Support**: PatchCore, EfficientAD, PaDiM anomaly detection
- **Data-Driven Thresholds**: Calibrated PASS/REVIEW/FAIL thresholds from validation data
- **Temporal Validation**: Frame-level persistence and duplicate-event suppression for video
- **Standardized Events**: CVEvent schema compatible with Safety Officer and Production Manager roles
- **REST API**: FastAPI endpoint for real-time inspection
- **Evidence Generation**: Automatic saving of inspection images and anomaly heatmaps

## Project Structure

```
quality_inspection/
├── api/                    # FastAPI inference endpoint
│   ├── __init__.py
│   └── main.py
├── configs/                 # Model and pipeline configurations
│   ├── patchcore.yaml
│   ├── efficientad.yaml
│   ├── padim.yaml
│   └── roi.yaml
├── data/                    # Dataset storage
│   ├── download/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/                  # Trained model checkpoints
│   ├── patchcore/
│   ├── efficientad/
│   └── padim/
├── results/                 # Evaluation results and artifacts
│   ├── metrics/
│   ├── visualizations/
│   ├── heatmaps/
│   ├── reports/
│   └── inspections/
├── scripts/                 # Execution scripts
│   ├── prepare_data.py
│   ├── train_patchcore.py
│   ├── train_efficientad.py
│   ├── train_padim.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── compare_models.py
│   ├── calibrate_thresholds.py
│   └── test_api.py
├── src/                     # Source code
│   ├── data/               # Dataset and dataloaders
│   ├── models/             # Anomaly detection models
│   ├── preprocessing/      # Image preprocessing
│   ├── roi/                # ROI extraction and pipeline
│   ├── inference/          # Inference engine
│   ├── postprocessing/     # Decision engine, temporal filter, localization
│   ├── evaluation/         # Metrics and comparison
│   ├── visualization/      # Plotting and visualization
│   └── events/             # CVEvent schema
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10+
- Consumer GPU with CUDA (optional, CPU fallback available)
- 8GB+ RAM recommended

### Setup

```bash
cd quality_inspection
pip install -r requirements.txt
```

## Dataset

### Primary Dataset: AutoVI

The Automotive Visual Inspection Dataset (AutoVI) is used for training and evaluation.

**Source**: https://autovi.utc.fr/
**Dataset**: https://zenodo.org/records/10459003

### Dataset Structure

```
data/raw/autovi/bracket_assembly/
├── train/
│   └── good/              # Nominal training images
├── test/
│   ├── good/              # Nominal test images
│   ├── bent_edge/         # Defect type 1
│   ├── missing_clip/      # Defect type 2
│   ├── surface_scratch/   # Defect type 3
│   └── weld_spatter/      # Defect type 4
└── ground_truth/
    ├── bent_edge/         # Ground truth masks
    ├── missing_clip/
    ├── surface_scratch/
    └── weld_spatter/
```

## Quick Start

### 1. Train Models

```bash
# Train PatchCore (baseline)
python3 scripts/train_patchcore.py

# Train EfficientAD (fast inference)
python3 scripts/train_efficientad.py

# Train PaDiM (classical baseline)
python3 scripts/train_padim.py
```

### 2. Evaluate Models

```bash
# Evaluate PatchCore
python3 scripts/evaluate.py --model patchcore

# Evaluate EfficientAD
python3 scripts/evaluate.py --model efficientad

# Evaluate PaDiM
python3 scripts/evaluate.py --model padim
```

### 3. Compare Models

```bash
python3 scripts/compare_models.py
```

This generates comparison reports in `results/reports/`:
- `model_comparison.md` - Markdown table
- `model_comparison.csv` - CSV for analysis
- `model_comparison.json` - Structured JSON

### 4. Calibrate Decision Thresholds

```bash
python3 scripts/calibrate_thresholds.py --model patchcore
```

This generates `results/reports/decision_calibration.json` with calibrated PASS/REVIEW/FAIL thresholds.

### 5. Start API Server

```bash
cd quality_inspection
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test API

```bash
python3 scripts/test_api.py
```

## Model Performance

Based on AutoVI bracket_assembly category evaluation:

| Model     | AUROC  | F1     | Pixel AUROC | Latency (ms) | FPS  | VRAM (MB) |
|-----------|--------|--------|-------------|--------------|------|-----------|
| PatchCore | 1.0000 | 1.0000 | 0.9974      | 143.4        | 7.0  | 0.0       |
| PaDiM     | 1.0000 | 1.0000 | 0.9957      | 80.5         | 12.4 | 0.0       |
| EfficientAD| 0.7900 | 0.8333 | 0.9787      | 53.4         | 18.7 | 0.0       |

**Note**: Benchmarks reflect controlled AutoVI test-split experiments on development hardware (CPU). Not real production-line validation.

## API Usage

### Inspect Single Image

```bash
curl -X POST "http://localhost:8000/inspect/image" \
  -F "file=@test_image.png" \
  -F "camera_id=camera_1" \
  -F "product_id=bracket_123" \
  -F "save_evidence=true"
```

### Inspect Video Frame

```bash
curl -X POST "http://localhost:8000/inspect/frame" \
  -F "file=@frame_001.png" \
  -F "frame_idx=1" \
  -F "timestamp=0.033" \
  -F "camera_id=camera_1" \
  -F "product_id=bracket_123"
```

### Configure Pipeline

```bash
curl -X POST "http://localhost:8000/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "model_checkpoint": "models/patchcore/patchcore_checkpoint.pt",
    "decision_calibration_path": "results/reports/decision_calibration.json",
    "enable_temporal": false,
    "output_dir": "results/inspections"
  }'
```

## Decision Engine

The decision engine converts anomaly scores into three-tier quality decisions:

- **PASS**: `score < pass_threshold` - Component approved
- **REVIEW**: `pass_threshold <= score < fail_threshold` - Manual inspection required
- **FAIL**: `score >= fail_threshold` - Component rejected

Thresholds are calibrated using validation data to avoid arbitrary selection:

```python
from src.postprocessing.decision_engine import DecisionEngine

# Load calibrated thresholds
engine = DecisionEngine.load_calibration("results/reports/decision_calibration.json")

# Make decision
decision = engine.decide(anomaly_score=1.5)
# Returns: QualityDecision.REVIEW
```

## Temporal Validation

For video streams, temporal validation avoids generating alerts on every frame:

- **Minimum Persistence**: Requires N consecutive frames with anomalies
- **Duplicate Suppression**: Prevents re-triggering events in same location
- **Spatial Matching**: Uses IoU to match events across frames

```python
from src.postprocessing.temporal_filter import TemporalFilter, TemporalFilterConfig

config = TemporalFilterConfig(
    minimum_persistence_frames=3,
    max_cooldown_frames=30,
    spatial_similarity_threshold=0.5,
)

temporal_filter = TemporalFilter(config)
```

## CVEvent Schema

Quality events follow the standardized CVEvent schema for multi-role compatibility:

```json
{
  "event_id": "uuid",
  "camera_id": "camera_1",
  "role": "quality",
  "task": "anomaly_inspection",
  "timestamp": "2026-10-01T10:00:00Z",
  "product_id": "bracket_123",
  "anomaly_score": 1.5,
  "decision": "REVIEW",
  "severity": "medium",
  "bbox": [x, y, w, h],
  "track_id": null,
  "evidence_path": "/path/to/evidence.png",
  "heatmap_path": "/path/to/heatmap.png",
  "metadata": {}
}
```

## Configuration

### Model Configuration

Edit YAML files in `configs/`:

```yaml
# configs/patchcore.yaml
model:
  backbone: "resnet18"
  layers: ["layer2", "layer3"]
  coreset_sampling_ratio: 0.10

dataset:
  category: "bracket_assembly"
  image_size: [256, 256]
  train_batch_size: 8
```

### ROI Configuration

Configure manual ROI in `configs/roi.yaml`:

```yaml
roi:
  type: "manual"
  x: 100
  y: 100
  width: 400
  height: 400
```

## Hardware Constraints

The system is designed for consumer laptop GPUs:

- **Memory**: Configurable batch sizes (default: 8)
- **Compute**: CPU fallback when GPU unavailable
- **Precision**: FP32 by default, mixed precision optional
- **VRAM**: Efficient models (ResNet18 backbone)

## Experiment Tracking

All experiments are logged to `results/metrics/experiment_log.jsonl`:

```json
{
  "timestamp": "2026-10-01T10:00:00Z",
  "dataset_version": "autovi",
  "category": "bracket_assembly",
  "model": "patchcore",
  "backbone": "resnet18",
  "image_auroc": 1.0,
  "pixel_auroc": 0.9974,
  "mean_latency_ms": 143.4,
  "fps": 7.0
}
```

## Development Status

### Completed (Steps 1-13)

- ✅ Dataset analysis and visualization
- ✅ Data pipeline with AutoVI integration
- ✅ PatchCore baseline implementation
- ✅ EfficientAD implementation
- ✅ PaDiM implementation
- ✅ ROI pipeline (manual, YOLO-ready)
- ✅ Decision engine with calibrated thresholds
- ✅ Anomaly localization and visualization
- ✅ Temporal validation for video
- ✅ CVEvent schema integration
- ✅ Model comparison and evaluation
- ✅ FastAPI inference endpoint

### Future Enhancements

- [ ] YOLO-based automatic ROI detection
- [ ] Dashboard integration with project orchestrator
- [ ] Real-time video stream processing
- [ ] Multi-camera support
- [ ] Additional AutoVI categories
- [ ] Production deployment validation

## Important Notes

**This is a research/academic prototype.**

Distinguish between:
- Benchmark performance (controlled test set)
- Experimental performance (development environment)
- Simulated factory-video performance
- Real production deployment

Do not claim suitability for real automotive production without proper industrial validation.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
dataset:
  train_batch_size: 4
  eval_batch_size: 4
```

### Slow Inference

- Use EfficientAD for fastest inference (18.7 FPS on CPU)
- Reduce image size: `image_size: [224, 224]`
- Enable GPU if available

### Path Issues

Ensure paths in configs are relative to `quality_inspection/` directory:
```yaml
dataset:
  root_dir: "data/raw/autovi/bracket_assembly"
```

## References

- **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
- **EfficientAD**: Schlegl et al., "EfficientAD: Efficient Texture Anomaly Detection for Industrial Quality Control", WACV 2024
- **PaDiM**: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2021
- **AutoVI**: Automotive Visual Inspection Dataset, https://autovi.utc.fr/

## License

This is a college mini-project for educational purposes.
