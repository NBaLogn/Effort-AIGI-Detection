#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-11
# description: Raw file dataset class for Effort model fine-tuning with direct file processing

import glob
import logging
import os
import random
from copy import deepcopy

import cv2
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


class RawFileDataset(data.Dataset):
    """Dataset class for processing raw file structures directly without JSON metadata.

    Supports:
    - Standalone images in real/fake directories
    - Video frames in real/frames/video_name/ and fake/frames/video_name/ structures
    - Optional landmarks and masks directories
    """

    def __init__(
        self,
        config: dict,
        mode: str = "train",
        raw_data_root: str | None = None,
    ):
        """Initialize the RawFileDataset.

        Args:
            config: Configuration dictionary
            mode: "train" or "test" mode
            raw_data_root: Path to root directory containing real/ and fake/ folders
        """
        self.config = config
        self.mode = mode
        self.raw_data_root = raw_data_root

        # Validate raw data root
        if not self.raw_data_root:
            raise ValueError("raw_data_root must be provided for RawFileDataset")
        if not os.path.exists(self.raw_data_root):
            raise FileNotFoundError(
                f"Raw data root directory not found: {self.raw_data_root}",
            )

        # Configuration parameters
        self.resolution = config.get("resolution", 224)
        self.frame_num = config.get("frame_num", {}).get(mode, 1)
        self.with_mask = config.get("with_mask", False)
        self.with_landmark = config.get("with_landmark", False)
        self.use_data_augmentation = config.get("use_data_augmentation", True)
        self.compression = config.get("compression", "c23")

        # Label mapping
        self.label_dict = config.get(
            "label_dict",
            {
                "UADFV_Real": 0,
                "UADFV_Fake": 1,
                # Add other default labels as needed
            },
        )

        # Initialize data structures
        self.image_list = []
        self.label_list = []
        self.video_name_list = []

        # File extensions to look for
        self.image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]

        # Initialize transforms
        self.transform = self._init_data_aug_method()
        self.mean = config.get("mean", [0.48145466, 0.4578275, 0.40821073])
        self.std = config.get("std", [0.26862954, 0.26130258, 0.27577711])

        # Validate directory structure
        self._validate_directory_structure()

        # Discover and build dataset index
        self._discover_and_build_index()

        # Create data_dict for compatibility with existing trainer
        self.data_dict = {
            "image": self.image_list,
            "label": self.label_list,
        }

        logger.info(
            f"RawFileDataset initialized with {len(self.image_list)} samples for {mode} mode",
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

    def _validate_directory_structure(self):
        """Validate that the directory structure is compatible with RawFileDataset."""
        logger.info(f"Validating directory structure: {self.raw_data_root}")

        # Check if directory is readable
        if not os.access(self.raw_data_root, os.R_OK):
            raise PermissionError(f"Cannot read directory: {self.raw_data_root}")

        # Check for at least one expected subdirectory
        expected_dirs = ["real", "fake"]
        found_dirs = []
        for expected_dir in expected_dirs:
            dir_path = os.path.join(self.raw_data_root, expected_dir)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                found_dirs.append(expected_dir)

        if not found_dirs:
            raise ValueError(
                f"No valid subdirectories found in {self.raw_data_root}. "
                f"Expected at least one of: {expected_dirs}",
            )

        logger.info(f"Found valid subdirectories: {found_dirs}")

    def _discover_and_build_index(self):
        """Discover files and build dataset index with balanced class distribution."""
        logger.info(f"Discovering files in {self.raw_data_root}")

        # Collect samples from each class separately
        real_samples = []
        fake_samples = []

        # Process real and fake directories
        for label_dir in ["real", "fake"]:
            dir_path = os.path.join(self.raw_data_root, label_dir)
            if not os.path.exists(dir_path):
                logger.warning(f"Directory not found: {dir_path}")
                continue

            label = self._get_label_for_directory(label_dir)
            samples = self._process_directory_get_samples(dir_path, label, label_dir)

            if label == 0:  # real
                real_samples.extend(samples)
            else:  # fake
                fake_samples.extend(samples)

        if not real_samples and not fake_samples:
            raise ValueError(f"No valid images found in {self.raw_data_root}")

        # Interleave real and fake samples for balanced batches
        self._interleave_samples(real_samples, fake_samples)

        logger.info(
            f"Dataset contains {len(real_samples)} real and {len(fake_samples)} fake samples",
        )

    def _get_label_for_directory(self, dir_name: str) -> int:
        """Get label for directory name."""
        # Try to find matching label in label_dict
        for key, value in self.label_dict.items():
            if dir_name.lower() in key.lower():
                return value

        # Default labeling: real -> 0, fake -> 1
        if "real" in dir_name.lower():
            return 0
        if "fake" in dir_name.lower():
            return 1
        raise ValueError(f"Cannot determine label for directory: {dir_name}")

    def _process_directory_get_samples(
        self,
        dir_path: str,
        label: int,
        label_name: str,
    ) -> list:
        """Process a directory and return collected samples without adding to main lists."""
        samples = []

        # 1. Process standalone images
        samples.extend(
            self._process_standalone_images_get_samples(dir_path, label, label_name),
        )

        # 2. Process video frames
        frames_dir = os.path.join(dir_path, "frames")
        if os.path.exists(frames_dir):
            samples.extend(
                self._process_video_frames_get_samples(frames_dir, label, label_name),
            )

        return samples

    def _process_directory(self, dir_path: str, label: int, label_name: str):
        """Process a directory (real or fake) and its contents."""
        logger.debug(f"Processing directory: {dir_path}")

        # 1. Process standalone images
        self._process_standalone_images(dir_path, label, label_name)

        # 2. Process video frames
        frames_dir = os.path.join(dir_path, "frames")
        if os.path.exists(frames_dir):
            self._process_video_frames(frames_dir, label, label_name)

    def _process_standalone_images(self, dir_path: str, label: int, label_name: str):
        """Process standalone images in the directory."""
        logger.debug(f"Looking for standalone images in: {dir_path}")

        for ext in self.image_extensions:
            files = glob.glob(os.path.join(dir_path, ext))
            for file_path in files:
                # Skip if this is part of a video frame directory
                if "frames" in file_path.split(os.sep):
                    continue

                self.image_list.append(file_path)
                self.label_list.append(label)
                self.video_name_list.append(
                    f"{label_name}_standalone_{os.path.basename(file_path)}",
                )

        logger.debug(
            f"Found {len([f for f in self.image_list if f'{label_name}_standalone' in f[-50:]])} standalone images",
        )

    def _process_video_frames(self, frames_dir: str, label: int, label_name: str):
        """Process video frames in frames/ directory."""
        logger.debug(f"Processing video frames in: {frames_dir}")

        # Get all video directories
        video_dirs = [
            d
            for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
        ]

        for video_dir in video_dirs:
            video_path = os.path.join(frames_dir, video_dir)
            frame_files = []

            # Collect all frame files
            for ext in self.image_extensions:
                frame_files.extend(glob.glob(os.path.join(video_path, ext)))

            if not frame_files:
                logger.warning(f"No frame files found in video directory: {video_path}")
                continue

            # Sort frames numerically
            try:
                frame_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
                )
            except ValueError:
                # If numeric sorting fails, sort alphabetically
                frame_files.sort()

            # Limit to frame_num if specified
            if self.frame_num > 0 and len(frame_files) > self.frame_num:
                # Select frames evenly distributed throughout the video
                step = len(frame_files) // self.frame_num
                frame_files = [
                    frame_files[i] for i in range(0, len(frame_files), step)
                ][: self.frame_num]

            # Add frames to dataset
            for i, frame_path in enumerate(frame_files):
                self.image_list.append(frame_path)
                self.label_list.append(label)
                self.video_name_list.append(f"{label_name}_{video_dir}_frame_{i}")

        logger.debug(
            f"Found {len([f for f in self.image_list if f'{label_name}_' in f[-50:] and '_frame_' in f[-50:]])} video frames",
        )

    def _process_standalone_images_get_samples(
        self, dir_path: str, label: int, label_name: str
    ) -> list:
        """Process standalone images and return samples without adding to main lists."""
        samples = []

        for ext in self.image_extensions:
            files = glob.glob(os.path.join(dir_path, ext))
            for file_path in files:
                # Skip if this is part of a video frame directory
                if "frames" in file_path.split(os.sep):
                    continue

                samples.append(
                    (
                        file_path,
                        label,
                        f"{label_name}_standalone_{os.path.basename(file_path)}",
                    )
                )

        return samples

    def _process_video_frames_get_samples(
        self, frames_dir: str, label: int, label_name: str
    ) -> list:
        """Process video frames and return samples without adding to main lists."""
        samples = []

        # Get all video directories
        video_dirs = [
            d
            for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
        ]

        for video_dir in video_dirs:
            video_path = os.path.join(frames_dir, video_dir)
            frame_files = []

            # Collect all frame files
            for ext in self.image_extensions:
                frame_files.extend(glob.glob(os.path.join(video_path, ext)))

            if not frame_files:
                continue

            # Sort frames numerically
            try:
                frame_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
                )
            except ValueError:
                # If numeric sorting fails, sort alphabetically
                frame_files.sort()

            # Limit to frame_num if specified
            if self.frame_num > 0 and len(frame_files) > self.frame_num:
                # Select frames evenly distributed throughout the video
                step = len(frame_files) // self.frame_num
                frame_files = [
                    frame_files[i] for i in range(0, len(frame_files), step)
                ][: self.frame_num]

            # Add frames to samples
            for i, frame_path in enumerate(frame_files):
                samples.append(
                    (
                        frame_path,
                        label,
                        f"{label_name}_{video_dir}_frame_{i}",
                    )
                )

        return samples

    def _interleave_samples(self, real_samples: list, fake_samples: list):
        """Interleave real and fake samples for balanced batching."""
        # Shuffle both lists individually first
        random.shuffle(real_samples)
        random.shuffle(fake_samples)

        # Interleave samples to ensure balanced batches
        self.image_list = []
        self.label_list = []
        self.video_name_list = []

        # Use round-robin interleaving
        max_len = max(len(real_samples), len(fake_samples))
        real_idx, fake_idx = 0, 0

        for i in range(max_len * 2):
            # Add real sample if available
            if real_idx < len(real_samples):
                img_path, label, video_name = real_samples[real_idx]
                self.image_list.append(img_path)
                self.label_list.append(label)
                self.video_name_list.append(video_name)
                real_idx += 1

            # Add fake sample if available
            if fake_idx < len(fake_samples):
                img_path, label, video_name = fake_samples[fake_idx]
                self.image_list.append(img_path)
                self.label_list.append(label)
                self.video_name_list.append(video_name)
                fake_idx += 1

            # Break if we've added all samples
            if real_idx >= len(real_samples) and fake_idx >= len(fake_samples):
                break

    def _load_rgb(self, file_path: str) -> Image.Image:
        """Load an RGB image from file path and resize."""
        try:
            img = cv2.imread(file_path)
            if img is None:
                img = Image.open(file_path)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if img is None:
                    raise ValueError(f"Loaded image is None: {file_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_CUBIC,
            )
            return Image.fromarray(np.array(img, dtype=np.uint8))
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            raise

    def _load_mask(self, file_path: str) -> np.ndarray | None:
        """Load a binary mask image."""
        if file_path is None or not os.path.exists(file_path):
            return None

        try:
            mask = cv2.imread(file_path, 0)
            if mask is None:
                return np.zeros((self.resolution, self.resolution, 1))
            mask = cv2.resize(mask, (self.resolution, self.resolution)) / 255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        except Exception as e:
            logger.warning(f"Error loading mask {file_path}: {e}")
            return None

    def _load_landmark(self, file_path: str) -> np.ndarray | None:
        """Load 2D facial landmarks."""
        if file_path is None or not os.path.exists(file_path):
            return None

        try:
            landmark = np.load(file_path)
            if self.resolution != 256:
                landmark = landmark * (self.resolution / 256)
            return np.float32(landmark)
        except Exception as e:
            logger.warning(f"Error loading landmark {file_path}: {e}")
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
        image_path = self.image_list[index]
        label = self.label_list[index]

        try:
            # Load image
            image = self._load_rgb(image_path)
            image_np = np.array(image)

            # Get corresponding mask and landmark paths
            mask_path = None
            landmark_path = None

            if "frames" in image_path.split(os.sep):
                # For video frames, look for landmarks and masks
                frame_dir = os.path.dirname(image_path)
                video_dir = os.path.basename(frame_dir)
                parent_dir = os.path.dirname(frame_dir)

                # Check for landmarks directory
                landmarks_dir = os.path.join(parent_dir, "landmarks", video_dir)
                if os.path.exists(landmarks_dir):
                    frame_name = os.path.splitext(os.path.basename(image_path))[0]
                    landmark_path = os.path.join(landmarks_dir, f"{frame_name}.npy")

                # Check for masks directory
                masks_dir = os.path.join(parent_dir, "masks", video_dir)
                if os.path.exists(masks_dir):
                    frame_name = os.path.splitext(os.path.basename(image_path))[0]
                    mask_path = os.path.join(masks_dir, f"{frame_name}.png")

            # Load mask and landmark if needed
            mask = (
                self._load_mask(mask_path)
                if (self.mode == "train" and self.with_mask)
                else None
            )
            landmarks = (
                self._load_landmark(landmark_path) if self.with_landmark else None
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
            logger.error(f"Error processing item at index {index} ({image_path}): {e}")
            raise

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_list)

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

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "num_samples": len(self),
            "num_real": sum(1 for label in self.label_list if label == 0),
            "num_fake": sum(1 for label in self.label_list if label == 1),
            "image_list": self.image_list,
            "label_list": self.label_list,
            "video_name_list": self.video_name_list,
        }
