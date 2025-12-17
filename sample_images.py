"""Image Sampling Script for Deepfake Detection Dataset.

This script samples a specified number of images from 'ai' and 'nature' folders
across multiple methods in a deepfake detection dataset.

Usage:
    python sample_images.py --source_dir /path/to/dataset --sample_size 500

The script creates a new directory with '_sampled' suffix containing the same
structure but with sampled images.
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

# Constants
IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
SAMPLE_SIZE_DEFAULT = 500

# Create a logger for this module
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory recursively.

    Args:
        directory: Directory to search for images

    Returns:
        List of Path objects for image files
    """
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.rglob(ext))
    return images


def sample_images(images: list[Path], sample_size: int) -> list[Path]:
    """Randomly sample images from a list.

    Args:
        images: List of image paths
        sample_size: Number of images to sample

    Returns:
        List of sampled image paths
    """
    if len(images) <= sample_size:
        logger.warning(
            "Only %d images found, using all available images",
            len(images),
        )
        return images

    sampled = random.sample(images, sample_size)
    logger.info("Sampled %d images from %d available", sample_size, len(images))
    return sampled


def copy_images(sampled_images: list[Path], dest_dir: Path) -> None:
    """Copy sampled images to destination directory, handling duplicate names.

    Args:
        sampled_images: List of image paths to copy
        dest_dir: Destination directory
    """
    used_names = set()

    for img_path in sampled_images:
        # Get the relative path from the category directory (fake/real)
        # to preserve subdirectory structure in the filename
        parts = img_path.parts
        try:
            # Find the category directory (fake or real) in the path
            category_index = (
                parts.index("fake") if "fake" in parts else parts.index("real")
            )
            # Create a name that includes parent directories to avoid conflicts
            relative_parts = parts[category_index:]
            # Join with underscores to create a unique filename
            unique_name = "_".join(relative_parts)
            dest_path = dest_dir / unique_name
        except ValueError:
            # Fallback if category not found
            unique_name = img_path.name
            dest_path = dest_dir / unique_name

        # Handle duplicates by adding a suffix
        original_dest_path = dest_path
        counter = 1
        while dest_path in used_names:
            name_parts = unique_name.rsplit(".", 1)
            if len(name_parts) == 2:
                base_name, ext = name_parts
                dest_path = dest_dir / f"{base_name}_{counter}.{ext}"
            else:
                dest_path = dest_dir / f"{unique_name}_{counter}"
            counter += 1

        used_names.add(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_path)
        logger.debug("Copied %s to %s", img_path, dest_path)


def process_dataset(source_dir: Path, sample_size: int, dest_dir: Path) -> None:
    """Process a dataset with fake and real subfolders.

    Args:
        source_dir: Path to source directory containing fake and real subfolders
        sample_size: Number of images to sample per category
        dest_dir: Destination directory
    """
    for category in ["fake", "real"]:
        category_dir = source_dir / category
        if not category_dir.exists():
            logger.warning("Category directory %s does not exist", category_dir)
            continue

        images = find_images(category_dir)
        if not images:
            logger.warning("No images found in %s", category_dir)
            continue

        sampled_images = sample_images(images, sample_size)
        dest_category_dir = dest_dir / category
        copy_images(sampled_images, dest_category_dir)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sample images from deepfake detection dataset",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source dataset directory",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=SAMPLE_SIZE_DEFAULT,
        help=f"Number of images to sample per category (default: {SAMPLE_SIZE_DEFAULT})",
    )

    args = parser.parse_args()

    setup_logging()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        logger.error("Source directory %s does not exist", source_dir)
        return

    dest_dir = source_dir.parent / f"{source_dir.name}_sampled"
    dest_dir.mkdir(exist_ok=True)

    logger.info("Processing dataset from %s", source_dir)
    logger.info("Output directory: %s", dest_dir)
    logger.info("Sample size per category: %d", args.sample_size)

    # Process the dataset
    logger.info("Processing %s", source_dir.name)
    process_dataset(source_dir, args.sample_size, dest_dir)

    logger.info("Image sampling completed")


if __name__ == "__main__":
    main()
