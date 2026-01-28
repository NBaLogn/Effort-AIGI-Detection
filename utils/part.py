import random
import shutil
from pathlib import Path


def move_random_files(source_dir, dest_dir, count):
    # Convert strings to Path objects
    source = Path(source_dir)
    dest = Path(dest_dir)

    # 1. Validation
    if not source.exists() or not source.is_dir():
        print(f"Error: Source folder '{source_dir}' does not exist.")
        return

    # Create destination folder if it doesn't exist
    dest.mkdir(parents=True, exist_ok=True)

    # 2. Get list of files only (ignores subfolders)
    files = [f for f in source.iterdir() if f.is_file()]

    if not files:
        print("No files found in the source directory.")
        return

    # 3. Determine how many files to move
    # Ensures we don't try to move more files than exist
    num_to_move = min(len(files), count)

    # 4. Select random files
    files_to_move = random.sample(files, num_to_move)

    # 5. Move the files
    print(f"Moving {num_to_move} files to '{dest_dir}'...")
    for file_path in files_to_move:
        try:
            shutil.move(str(file_path), str(dest / file_path.name))
            print(f"Moved: {file_path.name}")
        except Exception as e:
            print(f"Failed to move {file_path.name}: {e}")

    print("Operation complete.")

# --- Configuration ---
if __name__ == "__main__":
    # SOURCE_FOLDER = "/Volumes/Crucial/AI/DATASETS/SAMPLED/U_DiFF_sampled_30k/T2I_HPS/val/real"
    # DESTINATION_FOLDER = "/Volumes/Crucial/AI/DATASETS/CelebA/real_30k"
    SOURCE_FOLDER = "/Volumes/Crucial/AI/DATASETS/CelebA/real_30k"
    DESTINATION_FOLDER = "/Volumes/Crucial/AI/DATASETS/SAMPLED/U_DiFF_sampled_30k/T2I_SDXL/val/real"
    NUMBER_OF_FILES = 223

    move_random_files(SOURCE_FOLDER, DESTINATION_FOLDER, NUMBER_OF_FILES)
