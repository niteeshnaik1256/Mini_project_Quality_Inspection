# Automotive Visual Inspection Dataset (AutoVI) & Secondary Benchmarks

## 1. Primary Dataset: AutoVI
* **Official Portal:** https://autovi.utc.fr/
* **Zenodo Repository:** https://zenodo.org/records/10459003
* **Origin & Domain:** Genuine automotive assembly line imagery captured at Stellantis manufacturing facilities under realistic industrial lighting, vibration, specular reflection, and component position variations.
* **Problem Formulation:** One-class industrial anomaly detection (AOI). The system trains exclusively on nominal (defect-free) parts and evaluates on unseen nominal parts and real anomalous components (e.g. missing clips, misaligned cables, surface scratches, incorrect fasteners).
* **Hardware Constraint Compliance:** Designed for single-category targeted inspection on consumer GPUs (e.g. RTX 3050 Laptop 4GB VRAM). The full multi-gigabyte corpus is NOT required to be loaded concurrently.

## 2. Directory Structure Specification (MVTec / Anomalib Standard)
The dataset directory must follow the standard industrial AD hierarchy:
```
data/raw/autovi/<category>/
├── train/
│   └── good/                     # Defect-free reference parts (strictly nominal)
│       ├── 000.png
│       └── ...
├── test/
│   ├── good/                     # Defect-free test parts (evaluates false positives)
│   │   ├── 000.png
│   │   └── ...
│   ├── <defect_type_1>/          # Anomalous components of type 1
│   │   ├── 000.png
│   │   └── ...
│   └── <defect_type_2>/          # Anomalous components of type 2
│       └── ...
└── ground_truth/
    ├── <defect_type_1>/          # Pixel-level binary masks (0: background, 255: defect)
    │   ├── 000_mask.png
    │   └── ...
    └── <defect_type_2>/
        └── ...
```

## 3. Secondary Benchmarks (For Validation & Comparison Only)
* **MVTec AD:** Benchmark dataset spanning various textures and objects (carpet, bottle, cable, capsule, hazelnut, metal nut, etc.). Note: MVTec AD is a general industrial benchmark, NOT an automotive-specific dataset.
* **MPDD:** Metal Parts Defect Dataset.
* **GDXray:** X-ray industrial inspection dataset (welds, castings).
* **CarDD:** Surface damage on external car bodies (scratches, dents).

## 4. Preventing Train/Test Data Leakage
In unsupervised / self-supervised industrial anomaly detection:
1. Training sets MUST contain ONLY nominal items (`train/good`).
2. No images or transforms of training samples are permitted in `test/good`.
3. Checksum (MD5/SHA256) duplicate verification is performed automatically by `src/data/inspector.py`.
