"""Image Sampling Script for Deepfake Detection Dataset.

This script samples a specified number of images from a source directory.
It preserves the directory structure information by flattening the relative path
into the filename (e.g., 'sub/dir/img.png' -> 'sub_dir_img.png').

Usage:
    python sample_images.py --source_dir /path/to/dataset --sample_size 500
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Constants
DEFAULT_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
DEFAULT_SAMPLE_SIZE = 500

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: If True, set log level to DEBUG, else INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_images(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all image files in a directory recursively.

    Args:
        directory: Directory to search for images.
        extensions: List of file extensions to include (without dot).

    Returns:
        List of Path objects for image files.
    """
    images = []
    for ext in extensions:
        # Support both lowercase and uppercase extensions
        images.extend(directory.rglob(f"*.{ext}"))
        images.extend(directory.rglob(f"*.{ext.upper()}"))
    return sorted(list(set(images)))  # remove duplicates if any and sort


def sample_images(images: List[Path], sample_size: int) -> List[Path]:
    """Randomly sample images from a list.

    Args:
        images: List of image paths.
        sample_size: Number of images to sample.

    Returns:
        List of sampled image paths.
    """
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


def generate_unique_filename(
    source_path: Path, source_root: Path, dest_dir: Path
) -> Path:
    """Generate a unique filename in the destination directory based on relative path.

    Args:
        source_path: Full path to the source image.
        source_root: Root directory to calculate relative path from.
        dest_dir: Destination directory.

    Returns:
        Path object for the destination file.
    """
    try:
        relative_path = source_path.relative_to(source_root)
    except ValueError:
        # Fallback if source_path is not relative to source_root (shouldn't happen)
        relative_path = Path(source_path.name)

    # Flatten the path: 'a/b/c.jpg' -> 'a_b_c.jpg'
    flat_name = "_".join(relative_path.parts)
    dest_path = dest_dir / flat_name

    # Handle duplicates
    if dest_path.exists():
        stem = dest_path.stem
        suffix = dest_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
            
    return dest_path


def copy_images(
    sampled_images: List[Path], source_root: Path, dest_dir: Path
) -> None:
    """Copy sampled images to destination directory.

    Args:
        sampled_images: List of image paths to copy.
        source_root: Root directory of the source dataset.
        dest_dir: Destination directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for img_path in sampled_images:
        try:
            dest_path = generate_unique_filename(img_path, source_root, dest_dir)
            shutil.copy2(img_path, dest_path)
            success_count += 1
            logger.debug("Copied: %s -> %s", img_path, dest_path)
        except Exception as e:
            logger.error("Failed to copy %s: %s", img_path, e)

    logger.info("Successfully copied %d images to %s", success_count, dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images from dataset flattening subdirectory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="Path to the output directory (default: source_dir_sampled)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of images to sample total (or per category if structured)",
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
        "-v", "--verbose",
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
        dest_dir = source_dir.parent / f"{source_dir.name}_sampled"

    logger.info("Source: %s", source_dir)
    logger.info("Destination: %s", dest_dir)
    
    # Check if user wants to process entire dir as one or treat subdirs as categories
    # The original script hardcoded 'fake' and 'real'. 
    # To be generic, we'll just find all images in source_dir and sample them.
    # If the user wants per-category sampling, they can run the script multiple times 
    # or we can look for specific subdirs. 
    # Since the prompt asked for refactoring and the original script did 
    # 'fake' and 'real' specifically, let's keep it robust:
    # We will search the whole tree. If 'fake' and 'real' folders exist in the root,
    # we can process them separately to ensure balanced classes if desired?
    # Actually, the original script did `process_dataset` which looped over fake/real.
    # A generic "sample random N images from this folder" is arguably more useful.
    # If the user points to `dataset/`, and it has `fake/` and `real/`, finding ALL images
    # and sampling N might bias towards the larger class.
    # Let's try to detect if we are at a root with 'fake'/'real' and preserve that logic if matches,
    # otherwise fallback to simple bulk sampling.
    
    subdirs = [x.name for x in source_dir.iterdir() if x.is_dir()]
    has_fake_real = "fake" in subdirs and "real" in subdirs
    
    if has_fake_real:
        logger.info("Detected 'fake' and 'real' subdirectories. Sampling separately.")
        categories = ["fake", "real"]
    else:
        # Treat the source_dir as the single category
        categories = ["."]

    total_copied = 0
    for cat in categories:
        if cat == ".":
            cat_source = source_dir
            cat_dest = dest_dir
        else:
            cat_source = source_dir / cat
            cat_dest = dest_dir / cat
            
        logger.info("Processing category: %s", cat)
        images = find_images(cat_source, args.extensions)
        
        if not images:
            logger.warning("No images found in %s", cat_source)
            continue
            
        sampled = sample_images(images, args.sample_size)
        copy_images(sampled, cat_source, cat_dest)
        total_copied += len(sampled)

    logger.info("Completed. Total images copied: %d", total_copied)


if __name__ == "__main__":
    main()
