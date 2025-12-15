#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-12
# description: LMDB dataset class for Effort model training and evaluation

import logging
import os
import random
from copy import deepcopy

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

try:
    import albumentations as A
except ImportError:
    A = None

# Set up logging
logger = logging.getLogger(__name__)


class LMDBDataset(data.Dataset):
    """Dataset class for processing LMDB databases directly without JSON metadata.

    Supports LMDB databases created by convert_to_lmdb.py with the following key structure:
    - real/filename.jpg: Real images
    - fake/filename.jpg: Fake images
    - real/frames/video_name/frame.jpg: Video frames for real videos
    - fake/frames/video_name/frame.jpg: Video frames for fake videos
    - real/landmarks/video_name/frame.npy: Landmarks for real video frames
    - fake/landmarks/video_name/frame.npy: Landmarks for fake video frames
    - real/masks/video_name/frame.png: Masks for real video frames
    - fake/masks/video_name/frame.png: Masks for fake video frames
    """

    def __init__(
        self,
        config: dict,
        mode: str = "train",
        lmdb_path: str | None = None,
    ):
        """Initialize the LMDBDataset.

        Args:
            config: Configuration dictionary
            mode: "train" or "test" mode
            lmdb_path: Path to LMDB database directory
        """
        self.config = config
        self.mode = mode
        self.lmdb_path = lmdb_path or config.get("lmdb_path")

        if not self.lmdb_path:
            raise ValueError("lmdb_path must be provided in config or as parameter")
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        # Configuration parameters
        self.resolution = config.get("resolution", 224)
        self.frame_num = config.get("frame_num", {}).get(mode, 1)
        self.with_mask = config.get("with_mask", False)
        self.with_landmark = config.get("with_landmark", False)
        self.use_data_augmentation = config.get("use_data_augmentation", True)

        # Label mapping - hardcoded for LMDB datasets which use "real"/"fake" categories
        self.label_dict = {
            "real": 0,
            "fake": 1,
        }

        # Initialize data structures
        self.image_list = []
        self.label_list = []
        self.video_name_list = []

        # File extensions for different data types
        self.image_extensions = {".jpg", ".jpeg", ".png", ".gif"}

        # Initialize transforms
        self.transform = self._init_data_aug_method()
        self.mean = config.get("mean", [0.48145466, 0.4578275, 0.40821073])
        self.std = config.get("std", [0.26862954, 0.26130258, 0.27577711])

        # Open LMDB environment
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            subdir=True,
        )

        # Discover and build dataset index
        self._discover_and_build_index()

        # Create data_dict for compatibility with existing trainer
        self.data_dict = {
            "image": self.image_list,
            "label": self.label_list,
        }

        logger.info(
            f"LMDBDataset initialized with {len(self.image_list)} samples for {mode} mode from {self.lmdb_path}",
        )

    def _init_data_aug_method(self):
        """Initialize data augmentation pipeline."""
        if A is None:
            logger.warning("albumentations not available, using basic augmentation")
            # Return a simple identity transform
            return lambda x: x

        trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.HueSaturationValue(p=0.3),
                A.ImageCompression(quality_range=(40, 100), p=0.1),
                A.GaussNoise(p=0.1),
                A.MotionBlur(p=0.1),
                A.CLAHE(p=0.1),
                A.ChannelShuffle(p=0.1),
                A.CoarseDropout(p=0.1),
                A.RandomGamma(p=0.3),
                A.GlassBlur(p=0.3),
            ],
        )
        return trans

    def _discover_and_build_index(self):
        """Discover all keys in LMDB and build dataset index."""
        logger.info(f"Discovering keys in LMDB database: {self.lmdb_path}")

        # Collect all keys from LMDB
        all_keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                key = key_bytes.decode()
                all_keys.append(key)

        logger.info(f"Found {len(all_keys)} total keys in LMDB")

        # Group keys by data type and extract samples
        image_keys = [k for k in all_keys if self._is_image_key(k)]
        landmark_keys = [k for k in all_keys if self._is_landmark_key(k)]
        mask_keys = [k for k in all_keys if self._is_mask_key(k)]

        logger.info(f"Found {len(image_keys)} image keys")

        # Process image keys to build dataset
        processed_samples = set()  # Track processed samples to avoid duplicates

        for key in image_keys:
            sample_info = self._parse_image_key(key)
            if sample_info and sample_info["sample_id"] not in processed_samples:
                self.image_list.append(key)  # Store LMDB key instead of file path
                self.label_list.append(sample_info["label"])
                self.video_name_list.append(sample_info["video_name"])
                processed_samples.add(sample_info["sample_id"])

        logger.info(
            f"Dataset contains {len(self.image_list)} unique samples",
        )

    def _is_image_key(self, key: str) -> bool:
        """Check if key represents an image file."""
        return any(key.endswith(ext) for ext in self.image_extensions)

    def _is_landmark_key(self, key: str) -> bool:
        """Check if key represents a landmark file."""
        return key.endswith(".npy") and "landmarks" in key

    def _is_mask_key(self, key: str) -> bool:
        """Check if key represents a mask file."""
        return key.endswith(".png") and "masks" in key

    def _parse_image_key(self, key: str) -> dict | None:
        """Parse image key to extract label and metadata.

        Handles various LMDB key formats including:
        - fake/DATASETS/DFB/rgb/UADFV/real/frames/0000/000.png
        - fake/DATASETS/DFB/rgb/UADFV/fake/frames/0000_fake/000.png
        - real/filename.jpg (legacy format)
        - fake/filename.jpg (legacy format)
        """
        parts = key.split("/")

        # Skip non-image files and system files
        if not self._is_image_key(key) or ".DS_Store" in key:
            return None

        # Try to find category in the path
        category = None

        # Special handling for UADFV LMDB format: find category after "UADFV"
        if "UADFV" in parts:
            uadfv_index = parts.index("UADFV")
            if uadfv_index + 1 < len(parts):
                next_part = parts[uadfv_index + 1]
                if next_part.lower() in ["real", "fake"]:
                    category = next_part.lower()
        else:
            # Legacy format: look for first occurrence of "real" or "fake"
            for part in parts:
                if part.lower() in ["real", "fake"]:
                    category = part.lower()
                    break

        if category is None or category not in self.label_dict:
            logger.info(
                f"Skipping LMDB key {key}: category '{category}' not in label_dict {self.label_dict}",
            )
            return None

        label = self.label_dict[category]

        # Extract video/frame information
        if "frames" in parts:
            # Video frame format
            frames_index = parts.index("frames")
            if frames_index + 1 < len(parts):
                video_name = parts[frames_index + 1]
                frame_file = parts[-1]
                sample_id = f"{category}_{video_name}_{frame_file}"
            else:
                return None
        else:
            # Standalone image
            filename = parts[-1]
            video_name = f"{category}_standalone_{filename}"
            sample_id = video_name

        return {
            "label": label,
            "video_name": video_name,
            "sample_id": sample_id,
            "category": category,
        }

    def _load_rgb_from_lmdb(self, key: str) -> Image.Image:
        """Load an RGB image from LMDB."""
        try:
            with self.env.begin() as txn:
                image_bin = txn.get(key.encode())
                if image_bin is None:
                    raise ValueError(f"Image not found in LMDB: {key}")

                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image: {key}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_CUBIC,
            )
            return Image.fromarray(np.array(img, dtype=np.uint8))
        except Exception as e:
            logger.error(f"Error loading image {key}: {e}")
            raise

    def _load_mask_from_lmdb(self, key: str) -> np.ndarray | None:
        """Load a binary mask image from LMDB."""
        try:
            with self.env.begin() as txn:
                mask_bin = txn.get(key.encode())
                if mask_bin is None:
                    return None

                mask_buf = np.frombuffer(mask_bin, dtype=np.uint8)
                mask = cv2.imdecode(mask_buf, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    return None

                mask = cv2.resize(mask, (self.resolution, self.resolution)) / 255
                mask = np.expand_dims(mask, axis=2)
                return np.float32(mask)
        except Exception as e:
            logger.warning(f"Error loading mask {key}: {e}")
            return None

    def _load_landmark_from_lmdb(self, key: str) -> np.ndarray | None:
        """Load 2D facial landmarks from LMDB."""
        try:
            with self.env.begin() as txn:
                landmark_bin = txn.get(key.encode())
                if landmark_bin is None:
                    return None

                landmark = np.frombuffer(landmark_bin, dtype=np.uint32).reshape((81, 2))
                if self.resolution != 256:
                    landmark = landmark * (self.resolution / 256)
                return np.float32(landmark)
        except Exception as e:
            logger.warning(f"Error loading landmark {key}: {e}")
            return None

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert image to tensor."""
        return T.ToTensor()(img)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor."""
        normalize = T.Normalize(mean=self.mean, std=self.std)
        return normalize(img)

    def _data_aug(
        self,
        img: np.ndarray,
        landmark: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        augmentation_seed: int | None = None,
    ) -> tuple:
        """Apply data augmentation."""
        # Set seed for reproducibility
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        kwargs = {"image": img}
        if mask is not None:
            kwargs["mask"] = mask

        transformed = self.transform(**kwargs)

        augmented_img = transformed["image"]
        augmented_mask = transformed.get("mask")

        # Reset seeds
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, None, augmented_mask

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, np.ndarray | None, np.ndarray | None]:
        """Get item at index."""
        image_key = self.image_list[index]
        label = self.label_list[index]

        try:
            # Load image
            image = self._load_rgb_from_lmdb(image_key)
            image_np = np.array(image)

            # Get corresponding mask and landmark keys
            mask_key = None
            landmark_key = None

            # Convert image key to corresponding mask/landmark keys
            if "frames" in image_key:
                # For UADFV LMDB format: fake/DATASETS/DFB/rgb/UADFV/real/frames/0000/000.png
                # -> fake/DATASETS/DFB/rgb/UADFV/real/landmarks/0000/000.npy
                parts = image_key.split("/")
                # Find the category (real/fake) in the path
                category = None
                for part in parts:
                    if part.lower() in ["real", "fake"]:
                        category = part.lower()
                        break

                if category:
                    # Replace "frames" with "masks" or "landmarks"
                    mask_key = (
                        image_key.replace("/frames/", "/masks/")
                        .replace(".png", ".png")
                        .replace(".jpg", ".png")
                    )
                    landmark_key = (
                        image_key.replace("/frames/", "/landmarks/")
                        .replace(".png", ".npy")
                        .replace(".jpg", ".npy")
                    )

            # Load mask and landmark if needed
            mask = (
                self._load_mask_from_lmdb(mask_key)
                if (self.mode == "train" and self.with_mask and mask_key)
                else None
            )
            landmarks = (
                self._load_landmark_from_lmdb(landmark_key)
                if (self.with_landmark and landmark_key)
                else None
            )

            # Data augmentation
            if self.mode == "train" and self.use_data_augmentation:
                image_aug, landmarks_aug, mask_aug = self._data_aug(
                    image_np,
                    landmarks,
                    mask,
                )
            else:
                image_aug, landmarks_aug, mask_aug = (
                    deepcopy(image_np),
                    deepcopy(landmarks),
                    deepcopy(mask),
                )

            # Convert to tensor and normalize
            image_tensor = self._normalize(self._to_tensor(image_aug))

            return image_tensor, label, landmarks_aug, mask_aug

        except Exception as e:
            logger.error(f"Error processing item at index {index} ({image_key}): {e}")
            raise

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_list)

    def __getstate__(self):
        """Custom pickle state to exclude unpickleable objects."""
        logger.info("Excluding env and transform from LMDBDataset pickle state")
        state = self.__dict__.copy()
        # Exclude unpickleable objects
        if "env" in state:
            del state["env"]
        if "transform" in state:
            del state["transform"]
        return state

    def __setstate__(self, state):
        """Custom unpickle state to recreate unpickleable objects."""
        logger.info("Recreating env and transform in LMDBDataset worker process")
        self.__dict__.update(state)
        # Recreate LMDB environment
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            subdir=True,
        )
        # Recreate transform
        self.transform = self._init_data_aug_method()

    @staticmethod
    def collate_fn(batch: list[tuple]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        images, labels, landmarks, masks = zip(*batch)

        # Stack images and convert labels to tensor
        images_stacked = torch.stack(images, dim=0)
        labels_tensor = torch.LongTensor(labels)

        # Handle landmarks - stack if all are not None, else None
        if landmarks and not any(l is None for l in landmarks):
            landmarks_stacked = torch.stack(
                [
                    torch.from_numpy(l) if l is not None else torch.zeros((81, 2))
                    for l in landmarks
                ],
                dim=0,
            )
        else:
            landmarks_stacked = None

        # Handle masks - stack if all are not None, else None
        if masks and not any(m is None for m in masks):
            masks_stacked = torch.stack(
                [
                    torch.from_numpy(m) if m is not None else torch.zeros((1, 1, 1))
                    for m in masks
                ],
                dim=0,
            )
        else:
            masks_stacked = None

        return {
            "image": images_stacked,
            "label": labels_tensor,
            "landmark": landmarks_stacked,
            "mask": masks_stacked,
        }

    def close(self):
        """Close the LMDB environment."""
        if hasattr(self, "env"):
            self.env.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
