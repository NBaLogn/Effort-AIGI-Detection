#!/usr/bin/env python3
"""Split Dataset into Train/Val Directories.

This script splits a dataset organized as base_path/fake/ and base_path/real/
into train/val splits with a specified ratio. Files are randomly shuffled
and moved (not copied) to the new structure.

Input structure: base_path/fake/, base_path/real/
Output structure: base_path/train/fake|real/, base_path/val/fake|real/

Usage:
    python split_dataset_train_val.py /path/to/dataset
    # Default split ratio is 0.7 (70% train, 30% val)
"""

import os
import random
import shutil
from pathlib import Path


def split_dataset(base_path, split_ratio=0.7) -> None:
    """Split dataset into train/val directories with specified ratio.

    Args:
        base_path: Path to dataset root containing 'fake' and 'real' subdirectories
        split_ratio: Ratio for train split (default: 0.7 = 70% train, 30% val)
    """
    base_dir = Path(base_path)
    classes = ["fake", "real"]

    # Verify base directory exists
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist.")
        return

    # Create train and val directories
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    for cls in classes:
        # Source directory for the class
        src_cls_dir = base_dir / cls

        if not src_cls_dir.exists():
            print(f"Warning: Class directory {src_cls_dir} does not exist. Skipping.")
            continue

        files = [
            f
            for f in src_cls_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(
            f"Processing class '{cls}': Found {len(files)} files. Split: {len(train_files)} train, {len(val_files)} val."
        )

        # Create destination directories
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

        # Move files
        for f in train_files:
            shutil.move(str(f), str(train_dir / cls / f.name))

        for f in val_files:
            shutil.move(str(f), str(val_dir / cls / f.name))

        # Check if original directory is empty and remove it if so
        remaining_files = list(src_cls_dir.iterdir())
        if not remaining_files:
            src_cls_dir.rmdir()
            print(f"Removed empty source directory: {src_cls_dir}")
        else:
            print(
                f"Source directory {src_cls_dir} not empty (contains {len(remaining_files)} items). Kept.",
            )


if __name__ == "__main__":
    split_dataset(
        "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/facedata/ivansivkovenin_faces",
    )
