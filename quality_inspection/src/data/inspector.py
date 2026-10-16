"""
Industrial Dataset Inspector for Automotive Visual Inspection (AutoVI).
Analyzes one-class anomaly detection directory structures, validates ground-truth masks,
checks resolution consistency, computes defect distribution, and ensures zero train/test leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DatasetInspector")


@dataclass
class ImageResolutionStats:
    """Summary statistics for image resolutions in a split."""
    min_width: int = 0
    max_width: int = 0
    min_height: int = 0
    max_height: int = 0
    unique_resolutions: List[Tuple[int, int]] = field(default_factory=list)
    channel_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """Complete structural and statistical audit of an industrial inspection category."""
    category_name: str
    root_path: str
    train_normal_count: int
    test_normal_count: int
    test_anomaly_count: int
    total_samples: int
    defect_distribution: Dict[str, int]
    imbalance_ratio: float  # test_normal / test_anomaly
    resolution_stats: Dict[str, Any]
    mask_pairing_valid: bool
    unpaired_masks: List[str]
    train_test_leakage_detected: bool
    leakage_hashes: List[str]
    is_valid_anomalib_structure: bool
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetInspector:
    """
    Validates and audits industrial anomaly detection dataset splits.
    Ensures conformance with standard MVTec/Anomalib hierarchy:
      category/
        train/good/
        test/good/
        test/<defect_type>/
        ground_truth/<defect_type>/
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir).resolve()
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute MD5 hash to verify content identity without memory overhead."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect_category(self, category_name: Optional[str] = None) -> DatasetSummary:
        """
        Audit a specific category directory.
        If category_name is None, attempts to inspect dataset_dir itself or auto-discover subcategories.
        """
        cat_dir = self.dataset_dir / category_name if category_name else self.dataset_dir
        cat_name = category_name or cat_dir.name

        errors: List[str] = []

        train_dir = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        # 1. Structural Validation
        if not train_dir.exists():
            errors.append(f"Missing required training directory: {train_dir}")
        if not test_dir.exists():
            errors.append(f"Missing required test directory: {test_dir}")

        train_normal_files = [
            f for f in train_dir.glob("*") if f.suffix.lower() in self.IMAGE_EXTENSIONS
        ] if train_dir.exists() else []

        test_normal_dir = test_dir / "good"
        test_normal_files = [
            f for f in test_normal_dir.glob("*") if f.suffix.lower() in self.IMAGE_EXTENSIONS
        ] if test_normal_dir.exists() else []

        # 2. Defect categories in test/
        defect_distribution: Dict[str, int] = {}
        test_anomaly_files: List[Path] = []

        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                if sub.is_dir() and sub.name != "good":
                    files = [f for f in sub.glob("*") if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                    defect_distribution[sub.name] = len(files)
                    test_anomaly_files.extend(files)

        total_anomalies = len(test_anomaly_files)
        total_samples = len(train_normal_files) + len(test_normal_files) + total_anomalies

        # 3. Ground truth mask pairing check
        unpaired_masks: List[str] = []
        mask_pairing_valid = True

        if total_anomalies > 0:
            if not gt_dir.exists():
                errors.append(f"Ground truth directory missing: {gt_dir} despite having {total_anomalies} defect images.")
                mask_pairing_valid = False
            else:
                for defect_file in test_anomaly_files:
                    defect_type = defect_file.parent.name
                    expected_gt_dir = gt_dir / defect_type
                    stem = defect_file.stem
                    candidate_masks = [
                        expected_gt_dir / f"{stem}_mask{defect_file.suffix}",
                        expected_gt_dir / f"{stem}{defect_file.suffix}",
                        expected_gt_dir / f"{stem}_mask.png",
                        expected_gt_dir / f"{stem}.png",
                    ]
                    matched = any(p.exists() for p in candidate_masks)
                    if not matched:
                        unpaired_masks.append(str(defect_file.relative_to(cat_dir)))

                if unpaired_masks:
                    mask_pairing_valid = False
                    errors.append(f"Found {len(unpaired_masks)} anomalous test images lacking matching ground-truth masks.")

        # 4. Train / Test Data Leakage Check (MD5 Hash Collision)
        train_hashes: Dict[str, Path] = {}
        for f in train_normal_files:
            train_hashes[self._compute_file_hash(f)] = f

        leakage_hashes: List[str] = []
        for f in test_normal_files + test_anomaly_files:
            h = self._compute_file_hash(f)
            if h in train_hashes:
                leakage_hashes.append(f"Test file '{f.name}' matches Train file '{train_hashes[h].name}' (MD5: {h})")

        train_test_leakage_detected = len(leakage_hashes) > 0
        if train_test_leakage_detected:
            errors.append(f"DATA LEAKAGE DETECTED: {len(leakage_hashes)} identical images found between train and test splits!")

        # 5. Image Dimension and Channel Consistency
        resolutions: List[Tuple[int, int]] = []
        channels_map: Dict[int, int] = {}

        all_sample_files = (train_normal_files + test_normal_files + test_anomaly_files)[:100]  # Sample audit
        for img_path in all_sample_files:
            try:
                with Image.open(img_path) as img:
                    resolutions.append(img.size)  # (width, height)
                    ch = len(img.getbands())
                    channels_map[ch] = channels_map.get(ch, 0) + 1
            except Exception as e:
                errors.append(f"Corrupt image {img_path}: {e}")

        if resolutions:
            widths = [w for w, _ in resolutions]
            heights = [h for _, h in resolutions]
            res_stats = ImageResolutionStats(
                min_width=min(widths),
                max_width=max(widths),
                min_height=min(heights),
                max_height=max(heights),
                unique_resolutions=list(set(resolutions)),
                channel_counts=channels_map,
            )
        else:
            res_stats = ImageResolutionStats()

        # 6. Imbalance Ratio
        imbalance_ratio = (
            round(len(test_normal_files) / total_anomalies, 3)
            if total_anomalies > 0 else float("inf")
        )

        is_valid = len(errors) == 0 and len(train_normal_files) > 0

        return DatasetSummary(
            category_name=cat_name,
            root_path=str(cat_dir),
            train_normal_count=len(train_normal_files),
            test_normal_count=len(test_normal_files),
            test_anomaly_count=total_anomalies,
            total_samples=total_samples,
            defect_distribution=defect_distribution,
            imbalance_ratio=imbalance_ratio,
            resolution_stats=asdict(res_stats),
            mask_pairing_valid=mask_pairing_valid,
            unpaired_masks=unpaired_masks,
            train_test_leakage_detected=train_test_leakage_detected,
            leakage_hashes=leakage_hashes,
            is_valid_anomalib_structure=is_valid,
            validation_errors=errors,
        )

    def print_report(self, summary: DatasetSummary) -> None:
        """Render a formatted industrial inspection report to console."""
        print("=" * 70)
        print(f"  AUTOMOTIVE AOI DATASET AUDIT REPORT: [{summary.category_name.upper()}]")
        print("=" * 70)
        print(f"Path:                     {summary.root_path}")
        print(f"Valid Anomalib Schema:    {'[YES]' if summary.is_valid_anomalib_structure else '[NO]'}")
        print(f"Train/Test Leakage:       {'[VIOLATION DETECTED]' if summary.train_test_leakage_detected else '[CLEAN - No Leakage]'}")
        print(f"Mask Pairing Integrity:   {'[PASSED]' if summary.mask_pairing_valid else '[FAILED]'}")
        print("-" * 70)
        print("SAMPLE SPLIT COUNTS:")
        print(f"  * Train (Nominal Only):  {summary.train_normal_count} images")
        print(f"  * Test Nominal (Good):   {summary.test_normal_count} images")
        print(f"  * Test Anomalies:        {summary.test_anomaly_count} images")
        print(f"  * Total Samples:         {summary.total_samples} images")
        print(f"  * Test Imbalance (N/A):  {summary.imbalance_ratio} : 1")
        print("-" * 70)
        print("DEFECT DISTRIBUTION IN TEST SET:")
        if summary.defect_distribution:
            for defect_type, count in summary.defect_distribution.items():
                pct = (count / summary.test_anomaly_count * 100) if summary.test_anomaly_count > 0 else 0
                print(f"  - {defect_type:20s}: {count:4d} images ({pct:5.1f}%)")
        else:
            print("  (No anomalous test partitions detected)")
        print("-" * 70)
        print("RESOLUTION & ENCODING STATISTICS:")
        res = summary.resolution_stats
        print(f"  * Width Range:           {res.get('min_width')}px to {res.get('max_width')}px")
        print(f"  * Height Range:          {res.get('min_height')}px to {res.get('max_height')}px")
        print(f"  * Unique Resolutions:    {res.get('unique_resolutions')}")
        print(f"  * Color Channels:        {res.get('channel_counts')}")
        print("-" * 70)
        if summary.validation_errors:
            print("VALIDATION NOTICES / ERRORS:")
            for err in summary.validation_errors:
                print(f"  [!] {err}")
            print("-" * 70)
        print("=" * 70)

    def save_summary(self, summary: DatasetSummary, output_json_path: str | Path) -> None:
        """Export audit report to structured JSON for audit logging."""
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Audit summary written to {out_path}")
