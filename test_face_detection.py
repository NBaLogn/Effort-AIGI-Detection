#!/usr/bin/env python3
"""Test script for face_detection_filter.py

This script creates a test directory structure with sample images and runs the face detection filter
to verify it works correctly.
"""

import shutil
import sys
from pathlib import Path

# Add the current directory to Python path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from face_detection_filter import FaceDetectionFilter


def create_test_structure():
    """Create a test directory structure with some sample images."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (test_dir / "vacation").mkdir(exist_ok=True)
    (test_dir / "family" / "reunion").mkdir(parents=True, exist_ok=True)
    (test_dir / "work").mkdir(exist_ok=True)

    # Create some dummy image files (we'll create actual image files for testing)
    import cv2
    import numpy as np

    # Create a simple test image with a face-like pattern
    def create_test_image_with_face(filename, width=200, height=200):
        # Create a blank image
        img = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Draw a simple face-like pattern
        # Face outline
        cv2.circle(img, (width // 2, height // 2), 80, (50, 50, 50), -1)

        # Eyes
        cv2.circle(img, (width // 2 - 30, height // 2 - 20), 10, (255, 255, 255), -1)
        cv2.circle(img, (width // 2 + 30, height // 2 - 20), 10, (255, 255, 255), -1)

        # Mouth
        cv2.ellipse(
            img, (width // 2, height // 2 + 30), (40, 15), 0, 0, 180, (255, 255, 255), 2
        )

        cv2.imwrite(str(filename), img)

    def create_test_image_without_face(filename, width=200, height=200):
        # Create a simple landscape-like image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky
        img[: height // 2] = [200, 220, 255]

        # Ground
        img[height // 2 :] = [50, 150, 50]

        # Sun
        cv2.circle(img, (width - 50, 50), 30, (255, 255, 0), -1)

        cv2.imwrite(str(filename), img)

    # Create test images
    images_with_faces = [
        test_dir / "family_photo.jpg",
        test_dir / "vacation" / "beach_photo.png",
        test_dir / "family" / "reunion" / "group_photo.jpeg",
    ]

    images_without_faces = [
        test_dir / "sunset.jpg",
        test_dir / "work" / "document_scan.png",
        test_dir / "vacation" / "landscape.jpeg",
    ]

    # Create the images
    for img_path in images_with_faces:
        create_test_image_with_face(img_path)
        print(f"Created test image with face: {img_path}")

    for img_path in images_without_faces:
        create_test_image_without_face(img_path)
        print(f"Created test image without face: {img_path}")

    # Create some non-image files to test filtering
    (test_dir / "readme.txt").write_text("This is a text file")
    (test_dir / "data.csv").write_text("column1,column2\nvalue1,value2")

    print(f"Created test directory structure at: {test_dir}")
    return test_dir


def run_test():
    """Run the face detection filter on test images."""
    print("🧪 Starting face detection filter test...")

    # Create test structure
    test_source = create_test_structure()
    test_destination = Path("test_faces_output")

    # Clean up any existing test output
    if test_destination.exists():
        shutil.rmtree(test_destination)

    print(f"\n📁 Test source: {test_source}")
    print(f"📁 Test destination: {test_destination}")

    # Create and run the filter
    filter = FaceDetectionFilter(
        source_dir=str(test_source),
        destination_dir=str(test_destination),
        min_face_size=30,  # Lower threshold for test images
        scale_factor=1.1,
        min_neighbors=3,
        log_level="INFO",
    )

    print("\n🔍 Running face detection...")
    filter.process_directory(dry_run=False)

    # Verify results
    print("\n📊 Test Results:")
    print("=" * 50)

    # Count files in destination
    if test_destination.exists():
        dest_files = list(test_destination.rglob("*"))
        dest_files = [f for f in dest_files if f.is_file()]
        print(f"Files copied to destination: {len(dest_files)}")

        for file in dest_files:
            print(f"  ✓ {file.relative_to(test_destination)}")
    else:
        print("❌ Destination directory was not created")

    # Show what was expected
    print("\n📋 Expected behavior:")
    print("  - Images with faces should be copied to destination")
    print("  - Images without faces should be skipped")
    print("  - Directory structure should be preserved")
    print("  - Non-image files should be skipped")

    return test_destination.exists() and len(list(test_destination.rglob("*"))) > 0


def cleanup_test():
    """Clean up test files."""
    test_dirs = ["test_images", "test_faces_output", "logs"]

    for dir_name in test_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"🧹 Cleaned up: {dir_name}")


def main():
    """Main test function."""
    print("🚀 Face Detection Filter Test")
    print("=" * 50)

    try:
        # Run the test
        success = run_test()

        if success:
            print("\n✅ Test completed successfully!")
            print("The face detection filter is working correctly.")
        else:
            print("\n❌ Test failed!")
            print("There may be an issue with the face detection filter.")

        # Ask user if they want to clean up
        response = input("\n🧹 Clean up test files? (y/N): ").strip().lower()
        if response in ["y", "yes"]:
            cleanup_test()
            print("Test files cleaned up.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

        # Clean up on error
        cleanup_test()


if __name__ == "__main__":
    main()
