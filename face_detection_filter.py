#!/usr/bin/env python3
"""Face Detection Filter Script

Recursively processes all images in a source directory, detects faces using OpenCV Haar cascades,
and copies images containing faces to a destination directory while preserving the folder structure.

Usage:
    python face_detection_filter.py --source /path/to/source --destination /path/to/destination
    python face_detection_filter.py -s /Users/name/Pictures -d /Users/name/Faces_Only --min-size 50 --scale-factor 1.1 --min-neighbors 5
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2


class FaceDetectionFilter:
    """Main class for filtering images based on face detection."""

    # Supported image file extensions
    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }

    def __init__(
        self,
        source_dir: str,
        destination_dir: str,
        min_face_size: int = 50,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        log_level: str = "INFO",
    ):
        """Initialize the face detection filter.

        Args:
            source_dir: Source directory containing images
            destination_dir: Destination directory for filtered images
            min_face_size: Minimum face size to detect (in pixels)
            scale_factor: Scale factor for face detection
            min_neighbors: Minimum neighbors for face detection
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Statistics
        self.stats = {
            "total_images": 0,
            "images_with_faces": 0,
            "images_without_faces": 0,
            "errors": 0,
            "skipped_files": 0,
        }

        # Initialize logging
        self._setup_logging(log_level)

        # Load face detection model
        self.face_cascade = self._load_face_cascade()

        self.logger.info("Initialized FaceDetectionFilter:")
        self.logger.info(f"  Source: {self.source_dir}")
        self.logger.info(f"  Destination: {self.destination_dir}")
        self.logger.info(f"  Min face size: {self.min_face_size}px")
        self.logger.info(f"  Scale factor: {self.scale_factor}")
        self.logger.info(f"  Min neighbors: {self.min_neighbors}")

    def _setup_logging(self, log_level: str) -> None:
        """Set up comprehensive logging to file and console."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"face_detection_{timestamp}.log"

        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)

        # Create logger
        self.logger = logging.getLogger("FaceDetectionFilter")
        self.logger.setLevel(numeric_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging to: {log_file}")

    def _load_face_cascade(self) -> cv2.CascadeClassifier:
        """Load the Haar cascade classifier for face detection."""
        try:
            # Try to load the cascade from OpenCV's built-in data
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            if face_cascade.empty():
                raise ValueError("Failed to load face cascade classifier")

            self.logger.info("Successfully loaded Haar cascade classifier")
            return face_cascade

        except Exception as e:
            self.logger.error(f"Error loading face cascade: {e}")
            self.logger.error(
                "Make sure OpenCV is properly installed with cascade files"
            )
            sys.exit(1)

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if the file is an image based on its extension."""
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _detect_faces_in_image(self, image_path: Path) -> bool:
        """Detect faces in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            True if faces are detected, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))

            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            # Return True if faces are detected
            return len(faces) > 0

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return False

    def _copy_with_structure(self, source_path: Path, destination_path: Path) -> None:
        """Copy file to destination while preserving directory structure.

        Args:
            source_path: Source file path
            destination_path: Destination file path
        """
        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(source_path, destination_path)

        except Exception as e:
            self.logger.error(
                f"Error copying file {source_path} to {destination_path}: {e}"
            )
            raise

    def process_directory(self, dry_run: bool = False) -> None:
        """Process all images in the source directory recursively.

        Args:
            dry_run: If True, only log what would be done without actually copying files
        """
        self.logger.info("Starting directory processing...")
        self.logger.info(f"Source directory: {self.source_dir}")
        self.logger.info(f"Destination directory: {self.destination_dir}")
        self.logger.info(f"Dry run mode: {dry_run}")

        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            sys.exit(1)

        if not self.source_dir.is_dir():
            self.logger.error(f"Source path is not a directory: {self.source_dir}")
            sys.exit(1)

        # Process all files recursively
        for root, dirs, files in os.walk(self.source_dir):
            for file_name in files:
                file_path = Path(root) / file_name

                # Skip if not an image file
                if not self._is_image_file(file_path):
                    self.stats["skipped_files"] += 1
                    continue

                self.stats["total_images"] += 1

                # Check if file contains faces
                has_faces = self._detect_faces_in_image(file_path)

                if has_faces:
                    self.stats["images_with_faces"] += 1

                    # Calculate destination path preserving structure
                    relative_path = file_path.relative_to(self.source_dir)
                    destination_path = self.destination_dir / relative_path

                    if dry_run:
                        self.logger.info(
                            f"[DRY RUN] Would copy: {file_path} -> {destination_path}"
                        )
                    else:
                        try:
                            self._copy_with_structure(file_path, destination_path)
                            self.logger.debug(
                                f"Copied: {file_path} -> {destination_path}"
                            )
                        except Exception:
                            self.stats["errors"] += 1
                            continue

                else:
                    self.stats["images_without_faces"] += 1
                    self.logger.debug(f"No faces detected: {file_path}")

        self._print_summary()

    def _print_summary(self) -> None:
        """Print a summary of the processing results."""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total images processed: {self.stats['total_images']}")
        self.logger.info(f"Images with faces: {self.stats['images_with_faces']}")
        self.logger.info(f"Images without faces: {self.stats['images_without_faces']}")
        self.logger.info(f"Skipped files (non-images): {self.stats['skipped_files']}")
        self.logger.info(f"Errors encountered: {self.stats['errors']}")

        if self.stats["total_images"] > 0:
            face_detection_rate = (
                self.stats["images_with_faces"] / self.stats["total_images"]
            ) * 100
            self.logger.info(f"Face detection rate: {face_detection_rate:.1f}%")

        self.logger.info("=" * 60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Recursively filter images containing human faces and copy them to a destination directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection_filter.py -s /Users/name/Pictures -d /Users/name/Faces_Only
  python face_detection_filter.py -s /path/to/source -d /path/to/destination --min-size 80 --scale-factor 1.2
  python face_detection_filter.py -s /path/to/source -d /path/to/destination --dry-run
        """,
    )

    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="Source directory containing images to process",
    )
    parser.add_argument(
        "-d",
        "--destination",
        required=True,
        help="Destination directory for images with faces",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Minimum face size to detect in pixels (default: 50)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for face detection (default: 1.1)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for face detection (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    args = parser.parse_args()

    # Create and run the face detection filter
    filter = FaceDetectionFilter(
        source_dir=args.source,
        destination_dir=args.destination,
        min_face_size=args.min_size,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        log_level=args.log_level,
    )

    try:
        filter.process_directory(dry_run=args.dry_run)
        print("\n✅ Face detection filtering completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
