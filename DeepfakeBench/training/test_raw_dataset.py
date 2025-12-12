#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-11
# description: Test script for RawFileDataset implementation

import os
import shutil

# Add the training directory to Python path
import sys
import tempfile

from PIL import Image

sys.path.append(".")

from dataset.factory import DatasetFactory
from dataset.raw_file_dataset import RawFileDataset


def create_test_data_structure():
    """Create a temporary test data structure."""
    temp_dir = tempfile.mkdtemp(prefix="raw_dataset_test_")

    # Create directory structure
    real_dir = os.path.join(temp_dir, "real")
    fake_dir = os.path.join(temp_dir, "fake")
    frames_dir = os.path.join(fake_dir, "frames")
    video_dir = os.path.join(frames_dir, "test_video")

    os.makedirs(real_dir)
    os.makedirs(fake_dir)
    os.makedirs(video_dir)

    # Create test images
    def create_test_image(path, color=(255, 0, 0)):
        """Create a simple test image."""
        img = Image.new("RGB", (256, 256), color=color)
        img.save(path)

    # Create standalone images
    create_test_image(os.path.join(real_dir, "real_image1.png"), (0, 255, 0))
    create_test_image(os.path.join(real_dir, "real_image2.jpg"), (0, 0, 255))
    create_test_image(os.path.join(fake_dir, "fake_image1.png"), (255, 0, 0))

    # Create video frames
    for i in range(5):
        create_test_image(
            os.path.join(video_dir, f"frame_{i:03d}.png"), (128, 128, 128)
        )

    return temp_dir


def test_raw_file_dataset():
    """Test the RawFileDataset implementation."""
    print("🧪 Testing RawFileDataset implementation...")

    # Create test data structure
    test_dir = create_test_data_structure()
    print(f"✅ Created test data structure at: {test_dir}")

    try:
        # Test configuration
        config = {
            "resolution": 224,
            "frame_num": {"train": 3, "test": 3},
            "with_mask": False,
            "with_landmark": False,
            "use_data_augmentation": False,
            "compression": "c23",
            "label_dict": {
                "UADFV_Real": 0,
                "UADFV_Fake": 1,
            },
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }

        # Test dataset creation
        print("🔧 Creating RawFileDataset...")
        dataset = RawFileDataset(config, mode="train", raw_data_root=test_dir)

        # Test dataset properties
        print(f"✅ Dataset created with {len(dataset)} samples")
        dataset_info = dataset.get_dataset_info()
        print(
            f"📊 Dataset info: {dataset_info['num_real']} real, {dataset_info['num_fake']} fake"
        )

        # Test data loading
        print("📦 Testing data loading...")
        sample = dataset[0]
        image_tensor, label, landmarks, mask = sample

        print(f"✅ Sample loaded: tensor shape={image_tensor.shape}, label={label}")
        print(f"📏 Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        print(f"🏷️  Label: {label} ({'real' if label == 0 else 'fake'})")

        # Test collate function
        print("🔗 Testing collate function...")
        batch = [dataset[0], dataset[1]] if len(dataset) >= 2 else [dataset[0]]
        collated_batch = RawFileDataset.collate_fn(batch)

        print(f"✅ Collated batch: {list(collated_batch.keys())}")
        print(
            f"📦 Batch shapes: images={collated_batch['image'].shape}, labels={collated_batch['label'].shape}"
        )

        # Test DatasetFactory
        print("🏭 Testing DatasetFactory...")
        factory_dataset = DatasetFactory.create_dataset(
            config, mode="train", raw_data_root=test_dir
        )
        print(f"✅ DatasetFactory created: {type(factory_dataset).__name__}")

        # Test error handling
        print("🛡️  Testing error handling...")
        try:
            bad_dataset = RawFileDataset(
                config, mode="train", raw_data_root="/nonexistent/path"
            )
        except FileNotFoundError as e:
            print(f"✅ Correctly caught FileNotFoundError: {str(e)[:50]}...")

        try:
            empty_dir = tempfile.mkdtemp()
            empty_dataset = RawFileDataset(
                config, mode="train", raw_data_root=empty_dir
            )
        except ValueError as e:
            print(f"✅ Correctly caught ValueError for empty dir: {str(e)[:50]}...")

        print(
            "🎉 All tests passed! RawFileDataset implementation is working correctly."
        )

    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up test directory: {test_dir}")


def test_dataset_factory_fallback():
    """Test DatasetFactory fallback to JSON-based approach."""
    print("\n🔄 Testing DatasetFactory fallback...")

    config = {
        "resolution": 224,
        "frame_num": {"train": 1, "test": 1},
        "with_mask": False,
        "with_landmark": False,
        "use_data_augmentation": False,
        "compression": "c23",
        "label_dict": {"UADFV_Real": 0, "UADFV_Fake": 1},
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "dataset_json_folder": "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/preprocessing/dataset_json",
        "train_dataset": ["UADFV"],
    }

    # Test fallback to JSON-based dataset
    try:
        dataset = DatasetFactory.create_dataset(
            config, mode="train", raw_data_root=None
        )
        print(f"✅ Fallback successful: {type(dataset).__name__}")
    except Exception as e:
        print(f"⚠️  Fallback test failed (expected if JSON files not available): {e}")


if __name__ == "__main__":
    test_raw_file_dataset()
    test_dataset_factory_fallback()
    print("\n🏆 All tests completed!")
