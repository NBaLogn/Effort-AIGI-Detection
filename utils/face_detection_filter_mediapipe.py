#!/usr/bin/env python3
"""Filter Images Containing Faces using MediaPipe Face Detection.

This script recursively processes all images in a source directory, detects faces
using Google's MediaPipe face detection model, and copies only images containing
faces to a destination directory while preserving the folder structure.

MediaPipe is optimized for speed and works well on Apple Silicon (M1/M2) chips.
It's faster than RetinaFace but may be slightly less accurate.

Features:
- Fast face detection using MediaPipe BlazeFace model
- Optimized for Apple Silicon (MPS acceleration)
- Configurable detection confidence thresholds
- Comprehensive logging and progress tracking
- Recursive directory processing with structure preservation
- Parallel processing with configurable worker threads

Usage:
    uv run utils/face_detection_filter_mediapipe.py --source /path/to/source --destination /path/to/destination
    uv run utils/face_detection_filter_mediapipe.py -s /path/to/images -d /path/to/faces_only --confidence 0.9

See also:
    - face_detection_filter_retinaface.py: Higher accuracy, slower
    - face_detection_filter_yolo.py: Balanced speed/accuracy
"""

import argparse
import concurrent.futures
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from tqdm import tqdm


class MediaPipeFaceDetector:
    """Fast face detector using MediaPipe."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialize the MediaPipe face detector.

        Args:
            confidence_threshold: Minimum confidence score for face detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger("MediaPipeFaceDetector")

        # Download model if not present
        model_path = self._get_model_path()

        # Initialize MediaPipe Face Detector using the new API
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=confidence_threshold,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

        self.logger.info("MediaPipe Face Detection initialized")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")

    def _get_model_path(self) -> Path:
        """Download and return the path to the face detection model."""
        import urllib.request

        model_dir = Path.home() / ".mediapipe" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_detection_short_range.tflite"

        if not model_path.exists():
            self.logger.info("Downloading MediaPipe face detection model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(model_url, model_path)
            self.logger.info(f"Model downloaded to: {model_path}")

        return model_path

    def detect_faces(self, image) -> bool:
        """Detect faces in a single image.

        Args:
            image: Input image as numpy array (BGR format from cv2)

        Returns:
            True if faces are detected with sufficient confidence, False otherwise
        """
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Run face detection
            detection_result = self.detector.detect(mp_image)

            # Check if any faces were detected
            return len(detection_result.detections) > 0

        except Exception as e:
            self.logger.warning(f"Error during face detection: {e}")
            return False

    def get_face_count(self, image) -> int:
        """Get the number of faces detected.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Number of faces detected
        """
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            return len(detection_result.detections)

        except Exception as e:
            self.logger.warning(f"Error getting face count: {e}")
            return 0

    def __del__(self):
        """Clean up MediaPipe resources."""
        # New API handles cleanup automatically
        pass


class FaceDetectionFilter:
    """Main class for filtering images based on MediaPipe face detection."""

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
        confidence_threshold: float = 0.5,
        max_workers: int = 4,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the face detection filter.

        Args:
            source_dir: Source directory containing images
            destination_dir: Destination directory for filtered images
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
            max_workers: Number of worker threads for parallel processing
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers

        # Initialize logging first
        self._setup_logging(log_level)

        # Initialize face detector
        self.detector = MediaPipeFaceDetector(
            confidence_threshold=confidence_threshold,
        )

        # Statistics
        self.stats = {
            "total_images": 0,
            "images_with_faces": 0,
            "images_without_faces": 0,
            "errors": 0,
            "skipped_files": 0,
            "total_faces_detected": 0,
        }

        self.logger.info("Initialized FaceDetectionFilter (MediaPipe):")
        self.logger.info(f"  Source: {self.source_dir}")
        self.logger.info(f"  Destination: {self.destination_dir}")
        self.logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"  Max workers: {self.max_workers}")

    def _setup_logging(self, log_level: str) -> None:
        """Set up comprehensive logging to file and console."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"face_detection_mediapipe_{timestamp}.log"

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

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if the file is an image based on its extension."""
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _detect_faces_in_image(self, image_path: Path) -> tuple[bool, int]:
        """Detect faces in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (has_faces, face_count)
        """
        try:
            # Try reading with OpenCV first
            image = cv2.imread(str(image_path))

            # If OpenCV fails, try PIL as fallback
            if image is None:
                try:
                    self.logger.debug(f"OpenCV failed, trying PIL for: {image_path}")
                    pil_image = Image.open(image_path)

                    # Convert PIL image to OpenCV format (BGR)
                    # PIL images are RGB, OpenCV expects BGR
                    image_rgb = np.array(pil_image.convert("RGB"))
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    self.logger.debug(f"Successfully read with PIL: {image_path}")

                except Exception as pil_error:
                    self.logger.warning(
                        f"Could not read image with OpenCV or PIL: {image_path} - {pil_error}"
                    )
                    return False, 0

            # Detect faces
            face_count = self.detector.get_face_count(image)

            if face_count > 0:
                self.logger.debug(f"Faces detected in {image_path}: {face_count}")

            return face_count > 0, face_count

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return False, 0

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
                f"Error copying file {source_path} to {destination_path}: {e}",
            )
            raise

    def _process_single_file(self, file_path: Path, dry_run: bool) -> None:
        """Process a single file: detect faces and copy if found.

        Args:
            file_path: Path to the image file to process
            dry_run: Whether to perform a dry run
        """
        try:
            self.stats["total_images"] += 1

            # Check if file contains faces
            has_faces, face_count = self._detect_faces_in_image(file_path)

            if has_faces:
                self.stats["images_with_faces"] += 1
                self.stats["total_faces_detected"] += face_count

                # Calculate destination path preserving structure
                relative_path = file_path.relative_to(self.source_dir)
                destination_path = self.destination_dir / relative_path

                if dry_run:
                    self.logger.info(
                        f"[DRY RUN] Would copy: {file_path} -> {destination_path} (faces: {face_count})",
                    )
                else:
                    self._copy_with_structure(file_path, destination_path)
                    self.logger.debug(
                        f"Copied: {file_path} -> {destination_path} (faces: {face_count})",
                    )

            else:
                self.stats["images_without_faces"] += 1
                self.logger.debug(f"No faces detected: {file_path}")

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.exception(f"Error processing {file_path}: {e}")

    def process_directory(self, dry_run: bool = False) -> None:
        """Process all images in the source directory recursively.

        Args:
            dry_run: If True, only log what would be done without actually copying files
        """
        self.logger.info("Starting directory processing...")
        self.logger.info(f"Source directory: {self.source_dir}")
        self.logger.info(f"Destination directory: {self.destination_dir}")
        self.logger.info(f"Dry run mode: {dry_run}")
        self.logger.info(f"Parallel workers: {self.max_workers}")

        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            sys.exit(1)

        if not self.source_dir.is_dir():
            self.logger.error(f"Source path is not a directory: {self.source_dir}")
            sys.exit(1)

        # Collect all image files first
        image_files = []
        self.logger.info("Scanning for image files...")
        for root, _, files in os.walk(self.source_dir):
            for file_name in files:
                file_path = Path(root) / file_name
                if self._is_image_file(file_path):
                    image_files.append(file_path)
                else:
                    self.stats["skipped_files"] += 1

        total_files = len(image_files)
        self.logger.info(f"Found {total_files} images to process")

        # Process files in parallel
        # Note: MediaPipe is generally thread-safe for inference
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self._process_single_file, file_path, dry_run)
                for file_path in image_files
            ]

            # Use tqdm to track completion of futures
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=total_files,
                desc="Processing Images",
                unit="img",
            ):
                pass

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
        self.logger.info(f"Total faces detected: {self.stats['total_faces_detected']}")

        if self.stats["total_images"] > 0:
            face_detection_rate = (
                self.stats["images_with_faces"] / self.stats["total_images"]
            ) * 100
            self.logger.info(f"Face detection rate: {face_detection_rate:.1f}%")

        self.logger.info("=" * 60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Recursively filter images containing human faces using MediaPipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run face_detection_filter_mediapipe.py -s /Users/name/Pictures -d /Users/name/Faces_Only
  uv run face_detection_filter_mediapipe.py -s /path/to/source -d /path/to/destination --confidence 0.5
  uv run face_detection_filter_mediapipe.py -s /path/to/source -d /path/to/destination --workers 8
  uv run face_detection_filter_mediapipe.py -s /path/to/source -d /path/to/destination --dry-run
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
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for face detection (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    args = parser.parse_args()

    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)

    # Create and run the face detection filter
    face_filter = FaceDetectionFilter(
        source_dir=args.source,
        destination_dir=args.destination,
        confidence_threshold=args.confidence,
        max_workers=args.workers,
        log_level=args.log_level,
    )

    try:
        face_filter.process_directory(dry_run=args.dry_run)
        print("\n✅ Face detection filtering completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
