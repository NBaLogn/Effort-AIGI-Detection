"""Sample a fixed number of images from each subfolder.

This utility iterates over each immediate subdirectory of a dataset root and
copies up to `--sample_per_subfolder` randomly selected images from that subfolder
into a flat per-class output tree. Nested directories are collapsed inside the
filenames so that every class folder (`DEST/class/`) contains only files and no
deeper subdirectories. Files inherited from nested paths (for example
`class/fake/nestdir/image.jpg`) are renamed to `nestdir_image.jpg` before the copy.

Usage:
    uv run utils/sample_images_per_subfolder.py --source_dir /data --sample_per_subfolder 20
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

# Constants
DEFAULT_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
DEFAULT_SAMPLE_PER_SUBFOLDER = 25

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging verbosity."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_images(directory: Path, extensions: Iterable[str]) -> List[Path]:
    """Return all image files under `directory` matching the extensions."""

    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*.{ext}"))
        images.extend(directory.rglob(f"*.{ext.upper()}"))
    return sorted(set(images))


def sample_images(images: List[Path], sample_size: int) -> List[Path]:
    """Randomly select up to `sample_size` paths from `images`."""

    if not images:
        return []

    if len(images) <= sample_size:
        logger.info(
            "Subfolder has %d images (< sample size %d); copying all.",
            len(images),
            sample_size,
        )
        return images

    selected = random.sample(images, sample_size)
    logger.debug("Chosen %d images from %d available.", sample_size, len(images))
    return selected


def _resolve_collision(dest_path: Path) -> Path:
    """If a file already exists at dest_path, append a counter until unique."""

    if not dest_path.exists():
        return dest_path

    parent = dest_path.parent
    stem = dest_path.stem
    suffix = dest_path.suffix
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_images(sampled: List[Path], subfolder_root: Path, dest_subfolder: Path) -> int:
    """Copy sampled files into `dest_subfolder`, prefixing filenames with their immediate parent."""

    dest_subfolder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sampled:
        try:
            relative_path = src.relative_to(subfolder_root)
        except ValueError:
            logger.error("Image %s is outside source tree %s", src, subfolder_root)
            continue

        prefix = relative_path.parent.name or subfolder_root.name
        dest_name = f"{prefix}_{src.name}"
        dest_path = dest_subfolder / dest_name
        dest_path = _resolve_collision(dest_path)

        try:
            shutil.copy2(src, dest_path)
        except Exception as exc:
            logger.error("Failed to copy %s: %s", src, exc)
            continue
        copied += 1

    logger.debug("Copied %d files into %s", copied, dest_subfolder)
    return copied


def collect_subfolders(root: Path, allowed: Iterable[str] | None = None) -> List[Path]:
    """Return sorted list of immediate subdirectories, optionally filtering by name."""

    subfolders = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if allowed is None:
        return subfolders

    allowed_set = {name for name in allowed}
    filtered = [p for p in subfolders if p.name in allowed_set]
    missing = allowed_set - {p.name for p in filtered}
    for name in sorted(missing):
        logger.warning("Requested subfolder %s not found in %s", name, root)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample up to a fixed number of files from each subfolder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Root directory containing per-class subfolders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Destination root (defaults to <source>_sampled_by_subfolder)",
    )
    parser.add_argument(
        "--sample_per_subfolder",
        type=int,
        default=DEFAULT_SAMPLE_PER_SUBFOLDER,
        help="Number of images to sample from each immediate subfolder",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Image file extensions to include",
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        help="Optional list of specific subfolder names to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.sample_per_subfolder < 1:
        logger.error("Sample count per subfolder must be positive")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Seed set to %d", args.seed)

    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        logger.error("Source path does not exist: %s", source_dir)
        sys.exit(1)

    if args.output_dir:
        dest_root = args.output_dir.resolve()
    else:
        dest_root = source_dir.parent / f"{source_dir.name}_sampled_by_subfolder"

    logger.info("Source directory: %s", source_dir)
    logger.info("Destination directory: %s", dest_root)

    subfolders = collect_subfolders(source_dir, args.subfolders)
    if not subfolders:
        logger.error("No subfolders found under %s", source_dir)
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for subfolder in subfolders:
        logger.info("Processing %s", subfolder.name)
        images = find_images(subfolder, args.extensions)
        if not images:
            logger.warning("No images detected under %s", subfolder)
            continue

        sampled = sample_images(images, args.sample_per_subfolder)
        dest_subfolder = dest_root / subfolder.name
        copied = copy_images(sampled, subfolder, dest_subfolder)
        total_copied += copied

    logger.info("Completed sampling; copied %d images total", total_copied)


if __name__ == "__main__":
    main()
