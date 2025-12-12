"""Script to convert folders of image files to LMDB database.

Organizes data as:
- Standalone images: real/fake/standalone_filename.jpg
- Video frames: real/fake/frames/video_name/frame.jpg
- Landmarks: real/fake/landmarks/video_name/frame.npy
- Masks: real/fake/masks/video_name/frame.png

Author: Kilo Code
"""

import argparse
import logging
import sys
from pathlib import Path

import lmdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Real/fake folder name synonyms
REAL_SYNONYMS = {"real", "original", "nature"}
FAKE_SYNONYMS = {"fake", "ai", "synthesis", "synthetic", "manipulated"}

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
LANDMARK_EXTENSION = ".npy"
MASK_EXTENSION = ".png"

# Constants
MIN_DEPTH_FOR_VIDEO_DATA = 3


def get_category_from_path(path_parts: list[str]) -> str | None:
    """Determine if path indicates real or fake category."""
    for part in path_parts:
        part_lower = part.lower()
        if part_lower in REAL_SYNONYMS:
            return "real"
        if part_lower in FAKE_SYNONYMS:
            return "fake"
    return None


def calculate_map_size(input_folder: str) -> int:
    """Calculate estimated LMDB map size based on input data."""
    total_size = 0

    input_path = Path(input_folder)
    for file_path in input_path.rglob("*"):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except OSError:
                logger.warning("Could not get size for %s", file_path)

    # Estimate overhead: 2x for LMDB internal structures + 10% buffer
    estimated_size = int(total_size * 2.1)

    # Minimum 1GB, maximum reasonable limit
    min_size = 1024 * 1024 * 1024  # 1GB
    max_size = 100 * 1024 * 1024 * 1024  # 100GB

    map_size = max(min_size, min(estimated_size, max_size))

    logger.info("Estimated data size: %.2f GB", total_size / (1024**3))
    logger.info("Setting LMDB map size to: %.2f GB", map_size / (1024**3))

    return map_size


def process_file(
    file_path: str,
    category: str,
) -> tuple[str, bytes] | None:
    """Process a single file and return (key, binary_data) or None if unsupported."""
    try:
        # Get path relative to the category folder (real/ or fake/)
        file_path_obj = Path(file_path)

        # Find the category folder in the path
        parts = file_path_obj.parts
        category_index = -1
        for i, part in enumerate(parts):
            if part.lower() in REAL_SYNONYMS | FAKE_SYNONYMS:
                category_index = i
                break

        if category_index == -1:
            logger.warning("Could not find category folder in path: %s", file_path)
            return None

        # Get path relative to category folder
        category_folder = Path(*parts[: category_index + 1])
        rel_path = file_path_obj.relative_to(category_folder)

        # Determine the key structure based on the path
        rel_parts = rel_path.parts

        if len(rel_parts) == 1:
            # Standalone file directly in category folder
            key = f"{category}/{rel_parts[0]}"
        elif rel_parts[0] == "frames" and len(rel_parts) >= MIN_DEPTH_FOR_VIDEO_DATA:
            # Video frame: frames/video_name/frame_file
            video_name = rel_parts[1]
            frame_file = "/".join(rel_parts[2:])
            key = f"{category}/frames/{video_name}/{frame_file}"
        elif (
            rel_parts[0] in ("landmarks", "masks")
            and len(rel_parts) >= MIN_DEPTH_FOR_VIDEO_DATA
        ):
            # Landmark or mask: landmarks/video_name/file or masks/video_name/file
            data_type = rel_parts[0]
            video_name = rel_parts[1]
            data_file = "/".join(rel_parts[2:])
            key = f"{category}/{data_type}/{video_name}/{data_file}"
        else:
            # Other files, store with full relative path
            key = f"{category}/{'/'.join(rel_parts)}"

        # Read file as binary
        with file_path_obj.open("rb") as f:
            binary_data = f.read()
    except (OSError, ValueError):
        logger.exception("Error processing %s", file_path)
        return None
    else:
        return key, binary_data


def convert_to_lmdb(input_folder: str, output_lmdb: str) -> None:
    """Convert input folder to LMDB database."""
    input_path = Path(input_folder)
    if not input_path.exists():
        msg = f"Input folder not found: {input_folder}"
        raise FileNotFoundError(msg)

    # Calculate map size
    map_size = calculate_map_size(input_folder)

    # Create output directory if needed
    output_path = Path(output_lmdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s to LMDB at %s", input_folder, output_lmdb)

    processed_files = 0
    total_size = 0

    with (
        lmdb.open(str(output_path), map_size=map_size) as env,
        env.begin(write=True) as txn,
    ):
        for file_path in input_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Determine category from path
            path_parts = file_path.parts
            category = get_category_from_path(path_parts)
            if not category:
                continue  # Skip files not in real/fake folders

            # Process the file
            result = process_file(str(file_path), category)
            if result:
                key, binary_data = result
                txn.put(key.encode(), binary_data)
                processed_files += 1
                total_size += len(binary_data)

                if processed_files % 1000 == 0:
                    logger.info("Processed %d files...", processed_files)

    logger.info("Conversion complete!")
    logger.info("Processed %d files", processed_files)
    logger.info("Total data size: %.2f MB", total_size / (1024**2))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert folders of image files to LMDB database",
    )
    parser.add_argument(
        "input_folder",
        help="Path to input folder containing real/fake subfolders",
    )
    parser.add_argument(
        "output_lmdb",
        help="Path for output LMDB database file",
    )

    args = parser.parse_args()

    try:
        convert_to_lmdb(args.input_folder, args.output_lmdb)
        logger.info("Script completed successfully")
    except Exception:
        logger.exception("Script failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
