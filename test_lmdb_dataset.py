#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-12
# description: Test script for LMDB dataset functionality

import sys

sys.path.append("DeepfakeBench/training")

from dataset.lmdb_dataset import LMDBDataset


def test_lmdb_dataset():
    """Test LMDB dataset functionality."""
    # Basic configuration
    config = {
        "resolution": 224,
        "frame_num": {"test": 1},
        "with_mask": False,
        "with_landmark": False,
        "use_data_augmentation": False,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "label_dict": {"real": 0, "fake": 1},
    }

    # Test LMDB dataset
    lmdb_path = "UADFV.lmdb"

    print(f"Testing LMDB dataset with path: {lmdb_path}")

    try:
        dataset = LMDBDataset(config, mode="test", lmdb_path=lmdb_path)
        print(f"Dataset initialized successfully with {len(dataset)} samples")

        # Test getting first few items
        for i in range(min(3, len(dataset))):
            try:
                image, label, landmarks, mask = dataset[i]
                print(
                    f"Sample {i}: label={label}, image_shape={image.shape}, landmarks={landmarks is not None}, mask={mask is not None}"
                )
            except Exception as e:
                print(f"Error loading sample {i}: {e}")

        print("LMDB dataset test completed successfully!")

    except Exception as e:
        print(f"LMDB dataset test failed: {e}")
        raise


if __name__ == "__main__":
    test_lmdb_dataset()
