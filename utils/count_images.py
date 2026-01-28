import os


def count_images_recursive(directory):
    # Define the extensions to look for
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    count = 0

    # os.walk goes through every subfolder automatically
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                count += 1
    return count

# Replace with your path, e.g., "C:/Users/Photos" or "/Users/Name/Pictures"
target_folder = "/Volumes/Crucial/AI/DATASETS/Celeb-DF-v3/real"
total = count_images_recursive(target_folder)

print(f"Total images in all subfolders: {total}")
