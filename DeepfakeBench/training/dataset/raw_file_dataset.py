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
    - Flexible directory naming with synonyms and partial matches
    """

    # Directory name synonyms for real and fake classes
    REAL_SYNONYMS = [
        # Core terms
        "real",
        "original",
        "natural",
        # Variations with numbers
        "0_real",
        "real_0",
        "0real",
        "real0",
        "original_0",
        "natural_0",
        # Common prefixes/suffixes
        "0real",
        "real0",
        "real_",
        "_real",
        "original",
        "original_",
        "_original",
        "natural",
        "natural_",
        "_natural",
        # Alternative terms
        "authentic",
        "genuine",
        "true",
        # Case variations
        "Real",
        "REAL",
        "Original",
        "ORIGINAL",
        "Natural",
        "NATURAL",
        "Authentic",
        "AUTHENTIC",
        "Genuine",
        "GENUINE",
        "True",
        "TRUE",
        # Combined terms
        "real_images",
        "original_images",
        "natural_images",
        "real_data",
        "original_data",
        "natural_data",
        "real_samples",
        "original_samples",
        "natural_samples",
        "real_content",
        "original_content",
        "natural_content",
        "nature",
    ]

    FAKE_SYNONYMS = [
        # Core terms
        "fake",
        "synthetic",
        "ai",
        "generated",
        # Variations with numbers
        "1_fake",
        "fake_1",
        "1fake",
        "fake1",
        "1_synthetic",
        "synthetic_1",
        "1synthetic",
        "synthetic1",
        "1_ai",
        "ai_1",
        "1ai",
        "ai1",
        "1_generated",
        "generated_1",
        "1generated",
        "generated1",
        # Common prefixes/suffixes
        "fake_",
        "_fake",
        "synthetic_",
        "_synthetic",
        "ai_",
        "_ai",
        "generated_",
        "_generated",
        # Alternative terms
        "manipulated",
        "forged",
        "altered",
        "modified",
        "deepfake",
        "generated",
        "synthesized",
        # Case variations
        "Fake",
        "FAKE",
        "Synthetic",
        "SYNTHETIC",
        "AI",
        "Artificial",
        "Generated",
        "GENERATED",
        "Manipulated",
        "MANIPULATED",
        "Forged",
        "FORGED",
        "Altered",
        "ALTERED",
        "Modified",
        "MODIFIED",
        "Deepfake",
        "DEEPFAKE",
        "Synthesized",
        "SYNTHESIZED",
        # Combined terms
        "fake_images",
        "synthetic_images",
        "ai_images",
        "generated_images",
        "fake_data",
        "synthetic_data",
        "ai_data",
        "generated_data",
        "fake_samples",
        "synthetic_samples",
        "ai_samples",
        "generated_samples",
        "fake_content",
        "synthetic_content",
        "ai_content",
        "generated_content",
        "manipulated_content",
        "forged_content",
        "altered_content",
        # Common dataset patterns
        "fake",
        "fakes",
        "fake_images",
        "fake_videos",
        "synthetic",
        "synthetics",
        "synthetic_images",
        "synthetic_videos",
        "ai_generated",
        "ai_gen",
        "generated_ai",
        "deepfake",
        "deepfakes",
    ]

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

        # Auto-detect train/val splits if present
        if self.mode in ["train", "test", "val"]:
            mode_dir = "val" if self.mode == "test" else self.mode
            split_path = os.path.join(self.raw_data_root, mode_dir)

            # If the current root doesn't have real/fake but does have the split dir, move into it
            if os.path.exists(split_path) and os.path.isdir(split_path):
                # Check if current root has real/fake dirs
                has_direct_subdirs = False
                try:
                    current_subdirs = [
                        d
                        for d in os.listdir(self.raw_data_root)
                        if os.path.isdir(os.path.join(self.raw_data_root, d))
                    ]
                    for d in current_subdirs:
                        if self._is_real_directory(d) or self._is_fake_directory(d):
                            has_direct_subdirs = True
                            break
                except Exception:
                    pass

                if not has_direct_subdirs:
                    logger.info(
                        f"Auto-descending into {mode_dir} directory: {split_path}"
                    )
                    self.raw_data_root = split_path

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

        # Log all subdirectories found
        try:
            all_subdirs = [
                d
                for d in os.listdir(self.raw_data_root)
                if os.path.isdir(os.path.join(self.raw_data_root, d))
            ]
            logger.info(f"All subdirectories found: {all_subdirs}")
        except Exception as e:
            logger.warning(f"Could not list subdirectories: {e}")

        # Find directories that match our synonyms
        found_real_dirs = []
        found_fake_dirs = []

        for subdir in all_subdirs:
            if self._is_real_directory(subdir):
                found_real_dirs.append(subdir)
            elif self._is_fake_directory(subdir):
                found_fake_dirs.append(subdir)

        if not found_real_dirs and not found_fake_dirs:
            raise ValueError(
                f"No valid subdirectories found in {self.raw_data_root}. "
                f"Expected directories containing real/fake synonyms. "
                f"Found directories: {all_subdirs}. "
                f"Real synonyms: {self.REAL_SYNONYMS[:5]}... "
                f"Fake synonyms: {self.FAKE_SYNONYMS[:5]}...",
            )

        logger.info(f"Found real directories: {found_real_dirs}")
        logger.info(f"Found fake directories: {found_fake_dirs}")

    def _is_real_directory(self, dir_name: str) -> bool:
        """Check if directory name matches real synonyms."""
        dir_lower = dir_name.lower()

        # Exact matches
        real_synonyms_lower = [s.lower() for s in self.REAL_SYNONYMS]
        if dir_lower in real_synonyms_lower:
            return True

        # Partial matches - only for longer synonyms to avoid false positives (e.g., "ai" in "train")
        for synonym in self.REAL_SYNONYMS:
            syn_lower = synonym.lower()
            if len(syn_lower) > 3 and syn_lower in dir_lower:
                return True
            # For short synonyms, use word boundary check (simplified)
            if syn_lower in dir_lower.split("_") or syn_lower in dir_lower.split(" "):
                return True

        return False

    def _is_fake_directory(self, dir_name: str) -> bool:
        """Check if directory name matches fake synonyms."""
        dir_lower = dir_name.lower()

        # Exact matches
        fake_synonyms_lower = [s.lower() for s in self.FAKE_SYNONYMS]
        if dir_lower in fake_synonyms_lower:
            return True

        # Partial matches - only for longer synonyms to avoid false positives (e.g., "ai" in "train")
        for synonym in self.FAKE_SYNONYMS:
            syn_lower = synonym.lower()
            if len(syn_lower) > 3 and syn_lower in dir_lower:
                return True
            # For short synonyms, use word boundary check (simplified)
            if syn_lower in dir_lower.split("_") or syn_lower in dir_lower.split(" "):
                return True

        return False

    def _discover_and_build_index(self):
        """Discover files and build dataset index with balanced class distribution."""
        logger.info(f"Discovering files in {self.raw_data_root}")

        # Collect samples from each class separately
        real_samples = []
        fake_samples = []

        # Get all subdirectories
        all_subdirs = [
            d
            for d in os.listdir(self.raw_data_root)
            if os.path.isdir(os.path.join(self.raw_data_root, d))
        ]

        # Process each directory
        for subdir in all_subdirs:
            dir_path = os.path.join(self.raw_data_root, subdir)

            if self._is_real_directory(subdir):
                label = 0
                label_name = "real"
                logger.info(f"Processing real directory: {subdir}")
            elif self._is_fake_directory(subdir):
                label = 1
                label_name = "fake"
                logger.info(f"Processing fake directory: {subdir}")
            else:
                logger.debug(f"Skipping non-matching directory: {subdir}")
                continue

            samples = self._process_directory_get_samples(dir_path, label, label_name)

            if label == 0:  # real
                real_samples.extend(samples)
            else:  # fake
                fake_samples.extend(samples)

        if not real_samples and not fake_samples:
            raise ValueError(f"No valid images found in {self.raw_data_root}")

        # Interleave real and fake samples for balanced batches
        self._interleave_samples(real_samples, fake_samples)

        # Remove faulty images from the dataset
        self._remove_faulty_images()

        logger.info(
            f"Dataset contains {len(real_samples)} real and {len(fake_samples)} fake samples",
        )

    def _get_label_for_directory(self, dir_name: str) -> int:
        """Get label for directory name."""
        # Try to find matching label in label_dict
        for key, value in self.label_dict.items():
            if dir_name.lower() in key.lower():
                return value

        # Check our synonym lists
        if self._is_real_directory(dir_name):
            return 0
        if self._is_fake_directory(dir_name):
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
        self,
        dir_path: str,
        label: int,
        label_name: str,
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
                    ),
                )

        return samples

    def _process_video_frames_get_samples(
        self,
        frames_dir: str,
        label: int,
        label_name: str,
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
                    ),
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

    def _remove_faulty_images(self):
        """Remove images that cannot be loaded from the dataset."""
        logger.info("Checking for faulty images...")

        valid_indices = []
        faulty_count = 0

        for i, image_path in enumerate(self.image_list):
            try:
                # Try to load the image
                self._load_rgb(image_path)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Removing faulty image: {image_path} - {e}")
                faulty_count += 1

        if faulty_count > 0:
            # Filter out faulty images
            self.image_list = [self.image_list[i] for i in valid_indices]
            self.label_list = [self.label_list[i] for i in valid_indices]
            self.video_name_list = [self.video_name_list[i] for i in valid_indices]

            logger.info(
                f"Removed {faulty_count} faulty images. Remaining: {len(self.image_list)}",
            )
        else:
            logger.info("No faulty images found.")

    def _load_rgb(self, file_path: str) -> Image.Image:
        """Load an RGB image from file path and resize."""
        try:
            # Try OpenCV first
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_CUBIC,
                )
                return Image.fromarray(np.array(img, dtype=np.uint8))

            # If OpenCV fails, try PIL
            try:
                img = Image.open(file_path)
                img = img.convert("RGB")
                img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
                return img
            except Exception as pil_error:
                logger.error(f"Error loading image {file_path}: {pil_error}")
                raise ValueError(f"Cannot identify image file: {file_path}")

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
                logger.warning(f"Mask file exists but could not be read: {file_path}")
                return np.zeros((self.resolution, self.resolution, 1))
            mask = cv2.resize(mask, (self.resolution, self.resolution)) / 255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        except Exception as e:
            logger.warning(f"Error loading mask {file_path}: {e}")
            return np.zeros((self.resolution, self.resolution, 1))

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
        # Try to get a valid image, skipping faulty ones
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
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

                return image_tensor, label, landmarks_aug, mask_aug, image_path

            except Exception as e:
                logger.error(
                    f"Error processing item at index {index} ({image_path}): {e}",
                )
                attempt += 1

                if attempt >= max_attempts:
                    logger.error(
                        f"Failed to load valid image after {max_attempts} attempts, skipping this sample",
                    )
                    # Return a zero tensor as fallback
                    zero_image = torch.zeros(3, self.resolution, self.resolution)
                    return zero_image, label, None, None, "error_path"

                # Try next index
                index = (index + 1) % len(self.image_list)
                logger.warning(
                    f"Retrying with next sample (attempt {attempt}/{max_attempts})",
                )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch: list[tuple]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        images, labels, landmarks, masks, names = zip(*batch)

        # Stack images and convert labels to tensor
        images_stacked = torch.stack(images, dim=0)
        labels_tensor = torch.LongTensor(labels)

        # Handle landmarks - stack if all are not None, else None
        if landmarks and not any(l is None for l in landmarks):
            # Convert numpy arrays to tensors and handle None cases
            landmark_tensors = []
            for l in landmarks:
                if l is not None:
                    landmark_tensors.append(torch.from_numpy(l))
                else:
                    # Provide a zero tensor of correct shape if landmarks are missing for some items
                    landmark_tensors.append(torch.zeros((81, 2)))
            landmarks_stacked = torch.stack(landmark_tensors, dim=0)
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
            "name": list(names),
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


class MultiRawFileDataset(data.Dataset):
    """Dataset class that combines multiple RawFileDataset instances.

    This allows training/evaluation on multiple raw dataset directories simultaneously.
    """

    def __init__(
        self,
        config: dict,
        mode: str,
        raw_data_roots: list[str],
    ):
        """Initialize the MultiRawFileDataset.

        Args:
            config: Configuration dictionary
            mode: "train" or "test" mode
            raw_data_roots: List of paths to raw data directories
        """
        self.config = config
        self.mode = mode
        self.raw_data_roots = raw_data_roots

        # Create individual datasets
        self.datasets = []
        self.cumulative_lengths = [0]

        for root in raw_data_roots:
            dataset = RawFileDataset(config, mode, root)
            self.datasets.append(dataset)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

        # Create combined data_dict for compatibility with trainer
        self._build_data_dict()

        logger.info(
            f"MultiRawFileDataset initialized with {len(self.datasets)} datasets "
            f"and {len(self)} total samples",
        )

    def _build_data_dict(self):
        """Build combined data_dict from all sub-datasets for trainer compatibility."""
        all_images = []
        all_labels = []

        for dataset in self.datasets:
            all_images.extend(dataset.image_list)
            all_labels.extend(dataset.label_list)

        self.data_dict = {
            "image": all_images,
            "label": all_labels,
        }

    def __len__(self) -> int:
        """Get total dataset length."""
        return self.cumulative_lengths[-1]

    def __getitem__(self, index: int):
        """Get item at index across all datasets."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if index < cum_len:
                dataset_idx = i - 1
                break

        # Get the local index within that dataset
        local_index = index - self.cumulative_lengths[dataset_idx]

        try:
            img, label, landmark, mask, path = self.datasets[dataset_idx][local_index]
            return img, label, landmark, mask, path
        except Exception as e:
            logger.warning(
                f"Error getting item {index} from dataset {dataset_idx}: {e}",
            )
            # Return a zero tensor as fallback
            zero_image = torch.zeros(
                3,
                self.datasets[dataset_idx].resolution,
                self.datasets[dataset_idx].resolution,
            )
            return zero_image, 0, None, None, "error_path"

    @staticmethod
    def collate_fn(batch: list[tuple]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader - delegate to RawFileDataset."""
        return RawFileDataset.collate_fn(batch)

    def get_dataset_info(self) -> dict:
        """Get combined dataset information."""
        total_samples = 0
        total_real = 0
        total_fake = 0
        all_images = []
        all_labels = []
        all_video_names = []

        for i, dataset in enumerate(self.datasets):
            info = dataset.get_dataset_info()
            total_samples += info["num_samples"]
            total_real += info["num_real"]
            total_fake += info["num_fake"]
            all_images.extend(info["image_list"])
            all_labels.extend(info["label_list"])
            all_video_names.extend(info["video_name_list"])

        return {
            "num_samples": total_samples,
            "num_real": total_real,
            "num_fake": total_fake,
            "image_list": all_images,
            "label_list": all_labels,
            "video_name_list": all_video_names,
            "num_datasets": len(self.datasets),
        }
