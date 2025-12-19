"""Structure-Preserving Image Sampling Script.

This script samples a specified number of images from a source directory,
copying them to an output directory while PRESERVING the original folder structure.

Usage:
    uv run utils/sample_images_structured.py --source_dir /path/to/data --output_dir /path/to/output --sample_size 1000
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List

# Constants
DEFAULT_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
DEFAULT_SAMPLE_SIZE = 500

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_images(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all image files in a directory recursively."""
    images = []
    for ext in extensions:
        # Support both lowercase and uppercase extensions
        images.extend(directory.rglob(f"*.{ext}"))
        images.extend(directory.rglob(f"*.{ext.upper()}"))
    return sorted(list(set(images)))


def sample_images(images: List[Path], sample_size: int) -> List[Path]:
    """Randomly sample images from a list."""
    if not images:
        return []

    if len(images) <= sample_size:
        logger.warning(
            "Requested sample size (%d) >= available images (%d). Using all images.",
            sample_size,
            len(images),
        )
        return images

    sampled = random.sample(images, sample_size)
    logger.info("Sampled %d images from %d available.", sample_size, len(images))
    return sampled


def copy_images_preserving_structure(
    sampled_images: List[Path], source_root: Path, dest_root: Path
) -> None:
    """Copy images to destination maintaining relative paths."""
    success_count = 0

    for img_path in sampled_images:
        try:
            # Calculate relative path from source root
            # e.g. source/fake/img.png -> fake/img.png
            relative_path = img_path.relative_to(source_root)

            # Construct destination path
            dest_path = dest_root / relative_path

            # Ensure parent directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(img_path, dest_path)
            success_count += 1
            logger.debug("Copied: %s -> %s", img_path, dest_path)

        except ValueError:
            logger.error(
                "Path %s is not relative to source root %s", img_path, source_root
            )
        except Exception as e:
            logger.error("Failed to copy %s: %s", img_path, e)

    logger.info("Successfully copied %d images to %s", success_count, dest_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images from dataset while preserving directory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Path to the source dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output directory (default: source_dir_sampled_structured)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Total number of images to sample from the entire source directory",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="List of image extensions to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Random seed set to %d", args.seed)

    source_dir: Path = args.source_dir.resolve()
    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        sys.exit(1)

    # determine output directory
    if args.output_dir:
        dest_dir = args.output_dir.resolve()
    else:
        dest_dir = source_dir.parent / f"{source_dir.name}_sampled_structured"

    logger.info("Source: %s", source_dir)
    logger.info("Destination: %s", dest_dir)

    # 1. Find all images
    logger.info("Scanning for images...")
    all_images = find_images(source_dir, args.extensions)

    if not all_images:
        logger.error("No images found in %s", source_dir)
        sys.exit(0)

    logger.info("Found %d total images.", len(all_images))

    # 2. Sample images
    sampled_images = sample_images(all_images, args.sample_size)

    # 3. Copy preserving structure
    logger.info("Copying images...")
    copy_images_preserving_structure(sampled_images, source_dir, dest_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
