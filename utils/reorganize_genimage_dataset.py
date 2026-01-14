#!/usr/bin/env python3
"""Reorganize GenImageFaces Dataset Structure.

This script reorganizes the GenImageFaces dataset from a generator-based structure
to a standard train/val split structure. It flattens the directory hierarchy by:
- Moving files from Generator/split/class_name to split/mapped_class_name
- Mapping 'ai' -> 'fake' and 'nature' -> 'real'
- Prefixing filenames with generator name to prevent collisions
- Cleaning up empty directories after moving files

Input structure: GeneratorName/train|val/ai|nature/image.jpg
Output structure: train|val/fake|real/GeneratorName_image.jpg

Usage:
    uv run utils/reorganize_genimage_dataset.py /path/to/GenImageFaces --dry-run
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def reorganize_genimage_dataset(base_path, dry_run=False):
    """Reorganize GenImageFaces dataset from generator-based to split-based structure.

    Args:
        base_path: Path to the GenImageFaces dataset root directory
        dry_run: If True, only show what would be done without moving files

    """
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist.")
        return

    # Define the mapping from source to destination
    # Source structure: Generator/split/class_name
    # Destination structure: split/mapped_class_name
    # ai -> fake, nature -> real
    class_mapping = {"ai": "fake", "nature": "real"}

    # Identify generators (exclude existing train/val directories if they exist in root)
    # We iterate over directories that are not 'train' or 'val' (in case script is re-run or partially run)
    exclude_dirs = {"train", "val", ".DS_Store"}
    generators = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs
    ]

    if not generators:
        print("No generator directories found.")
        return

    print(f"Found {len(generators)} generators: {[g.name for g in generators]}")

    # Prepare destination directories
    splits = ["train", "val"]
    dest_classes = ["fake", "real"]

    for split in splits:
        for cls in dest_classes:
            dest_dir = base_dir / split / cls
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"Destination mapped: {dest_dir}")

    # Iterate and move files
    moved_count = 0

    for generator in tqdm(generators, desc="Processing Generators"):
        for split in splits:
            for src_cls, dest_cls in class_mapping.items():
                src_dir = generator / split / src_cls

                if not src_dir.exists():
                    # print(f"Skipping missing directory: {src_dir}")
                    continue

                dest_dir = base_dir / split / dest_cls

                # Get all files
                files = [
                    f
                    for f in src_dir.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]

                for f in files:
                    # Create a new unique filename to prevent collisions between generators
                    # e.g., ADM_0_adm_0.PNG
                    new_filename = f"{generator.name}_{f.name}"
                    dest_path = dest_dir / new_filename

                    if dry_run:
                        # print(f"[Dry Run] would move {f} -> {dest_path}")
                        pass
                    else:
                        shutil.move(str(f), str(dest_path))

                    moved_count += 1

                # Cleanup empty source directories
                if not dry_run:
                    try:
                        src_dir.rmdir()
                        # specific mapped dir (e.g. ai/nature) removed
                        # try removing split dir (train/val) if empty
                        if not any(generator.joinpath(split).iterdir()):
                            generator.joinpath(split).rmdir()
                    except OSError:
                        pass  # Directory not empty or other error

        # Try removing generator dir if empty
        if not dry_run:
            if not any(generator.iterdir()):
                try:
                    generator.rmdir()
                    print(f"Removed empty generator directory: {generator.name}")
                except OSError:
                    pass

    print(f"\nOperation completed. Moved {moved_count} files.")
    if dry_run:
        print("This was a DRY RUN. No files were actually moved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize GenImageFaces dataset from generator-based to split-based structure.",
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default="/Volumes/Crucial/AI/DATASETS/SAMPLED/df40",
        help="Path to the GenImageFaces dataset root",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without executing them",
    )

    args = parser.parse_args()

    print(f"Processing dataset at: {args.base_path}")
    if args.dry_run:
        print("!!! DRY RUN MODE !!!")

    reorganize_genimage_dataset(args.base_path, dry_run=args.dry_run)
