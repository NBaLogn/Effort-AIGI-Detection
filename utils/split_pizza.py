import os
import random
import shutil

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")


def _collect_image_paths(root_dir, valid_extensions):
    matches = []
    for current_root, dirs, files in os.walk(root_dir):
        # Avoid descending into the subsets we create so repeat runs remain idempotent
        dirs[:] = [d for d in dirs if not d.startswith("subset_")]
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                matches.append(os.path.join(current_root, filename))
    return matches


def _unique_destination(path):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def split_images_into_folders(source_dir, num_folders=22):
    images = _collect_image_paths(source_dir, VALID_EXTENSIONS)
    if not images:
        print("No images found in the source directory.")
        return

    random.shuffle(images)

    subset_dirs = []
    for i in range(num_folders):
        subset_path = os.path.join(source_dir, f"subset_{i+1}")
        os.makedirs(subset_path, exist_ok=True)
        subset_dirs.append(subset_path)

    for index, source_path in enumerate(images):
        destination_folder = subset_dirs[index % num_folders]
        destination_path = os.path.join(destination_folder, os.path.basename(source_path))
        destination_path = _unique_destination(destination_path)
        shutil.move(source_path, destination_path)

    print(f"Successfully distributed {len(images)} images into {num_folders} folders.")


if __name__ == "__main__":
    source_directory = "/Volumes/Crucial/AI/DATASETS/Celeb-DF-v3/real"
    split_images_into_folders(source_directory)
