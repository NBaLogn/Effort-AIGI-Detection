#!/usr/bin/env python3
"""Face Detection Filter Script using InsightFace RetinaFace.

Recursively processes all images in a source directory, detects faces using InsightFace RetinaFace,
and copies images containing faces to a destination directory while preserving the folder structure.

Features:
- High-accuracy face detection using RetinaFace (90.4% mAP on hard datasets)
- MPS (Metal Performance Shaders) acceleration for Mac M1/M2 chips
- Configurable detection confidence thresholds
- Comprehensive logging and progress tracking
- Recursive directory processing with structure preservation

Usage:
    python face_detection_filter_retinaface.py --source /path/to/source --destination /path/to/destination
    python face_detection_filter_retinaface.py -s /Users/name/Pictures -d /Users/name/Faces_Only --confidence 0.7
"""

import argparse
import concurrent.futures
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from tqdm import tqdm


class RetinaFaceDetector:
    """High-accuracy face detector using InsightFace with MPS support."""

    def __init__(self, confidence_threshold: float = 0.5, device: str | None = None):
        """Initialize the face detector using FaceAnalysis.

        Args:
            confidence_threshold: Minimum confidence score for face detection (0.0-1.0)
            device: Device to use ('mps', 'cpu', or None for auto-detection)
        """
        self.confidence_threshold = confidence_threshold

        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.logger = logging.getLogger("RetinaFaceDetector")
        self.logger.info(f"Initializing FaceAnalysis detector on {self.device}")

        # Load the model
        self.model = self._load_model()

        self.logger.info(
            f"FaceAnalysis detector initialized successfully on {self.device}",
        )
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")

    def _load_model(self):
        """Load the FaceAnalysis model with appropriate device configuration."""
        try:
            self.logger.debug("Attempting to load FaceAnalysis model...")

            # Determine providers based on device
            if self.device == "mps":
                # CoreML for Apple Silicon
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                # CPU only
                providers = ["CPUExecutionProvider"]

            self.logger.debug(f"Using providers: {providers}")

            # Load FaceAnalysis with detection model
            model = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allow_modules=["detection"],
            )

            # Prepare the detection model
            # ctx_id=0 means GPU/accelerated, but with CoreML provider it will use appropriate backend
            self.logger.debug(
                f"Preparing detection model with det_thresh={self.confidence_threshold}",
            )
            model.prepare(ctx_id=0, det_thresh=self.confidence_threshold)

            self.logger.debug("Model loaded and prepared successfully")
            return model

        except Exception as e:
            self.logger.exception(f"Failed to load FaceAnalysis model: {e}")
            self.logger.exception(
                "Make sure insightface is installed: pip install insightface",
            )
            raise

    def detect_faces(self, image: np.ndarray) -> bool:
        """Detect faces in a single image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            True if faces are detected with sufficient confidence, False otherwise
        """
        try:
            # Run face detection
            faces = self.model.get(image)

            # Check if any face meets the confidence threshold
            for face in faces:
                if hasattr(face, "det_score"):
                    confidence = float(face.det_score)
                else:
                    # Fallback for different face object structure
                    confidence = float(face[4]) if len(face) > 4 else 0.5

                if confidence >= self.confidence_threshold:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error during face detection: {e}")
            return False

    def get_face_count_and_confidences(
        self,
        image: np.ndarray,
    ) -> tuple[int, list[float]]:
        """Get the number of faces and their confidence scores.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Tuple of (face_count, confidence_scores)
        """
        try:
            faces = self.model.get(image)
            confidences = []

            for face in faces:
                if hasattr(face, "det_score"):
                    confidence = float(face.det_score)
                else:
                    confidence = float(face[4]) if len(face) > 4 else 0.0
                confidences.append(confidence)

            # Count faces above threshold
            face_count = sum(1 for c in confidences if c >= self.confidence_threshold)

            return face_count, confidences

        except Exception as e:
            self.logger.warning(f"Error getting face count: {e}")
            return 0, []


class FaceDetectionFilter:
    """Main class for filtering images based on RetinaFace detection."""

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
        device: str | None = None,
        max_workers: int = 4,
        log_level: str = "INFO",
    ):
        """Initialize the face detection filter.

        Args:
            source_dir: Source directory containing images
            destination_dir: Destination directory for filtered images
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
            device: Device to use ('mps', 'cpu', or None for auto-detection)
            max_workers: Number of worker threads for parallel processing
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.max_workers = max_workers

        # Initialize face detector
        self.detector = RetinaFaceDetector(
            confidence_threshold=confidence_threshold,
            device=device,
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

        # Initialize logging
        self._setup_logging(log_level)

        self.logger.info("Initialized FaceDetectionFilter:")
        self.logger.info(f"  Source: {self.source_dir}")
        self.logger.info(f"  Destination: {self.destination_dir}")
        self.logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"  Device: {self.device or 'auto-detected'}")
        self.logger.info(f"  Max workers: {self.max_workers}")

    def _setup_logging(self, log_level: str) -> None:
        """Set up comprehensive logging to file and console."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"face_detection_retinaface_{timestamp}.log"

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

    def _detect_faces_in_image(self, image_path: Path) -> tuple[bool, int, list[float]]:
        """Detect faces in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (has_faces, face_count, confidences)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))

            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False, 0, []

            # Detect faces
            has_faces, confidences = self.detector.get_face_count_and_confidences(image)
            face_count = sum(1 for c in confidences if c >= self.confidence_threshold)

            if face_count > 0:
                self.logger.debug(
                    f"Faces detected in {image_path}: {face_count} (confidences: {[f'{c:.3f}' for c in confidences]})",
                )

            return face_count > 0, face_count, confidences

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return False, 0, []

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
            has_faces, face_count, _ = self._detect_faces_in_image(
                file_path,
            )

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
            self.logger.error(f"Error processing {file_path}: {e}")

    def process_directory(self, dry_run: bool = False) -> None:
        """Process all images in the source directory recursively.

        Args:
            dry_run: If True, only log what would be done without actually copying files
        """
        self.logger.info("Starting directory processing...")
        self.logger.info(f"Source directory: {self.source_dir}")
        self.logger.info(f"Destination directory: {self.destination_dir}")
        self.logger.info(f"Dry run mode: {dry_run}")
        self.logger.info(f"Using device: {self.detector.device}")
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
        # Note: We use ThreadPoolExecutor because FaceAnalysis/ONNXRuntime often releases GIL
        # during inference, allowing for parallelism.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as executor:
            # Create a partial function or lambda if needed, but simple loop works with submit
            # We use list(tqdm(...)) to force iteration and display progress

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
        description="Recursively filter images containing human faces using InsightFace RetinaFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection_filter_retinaface.py -s /Users/name/Pictures -d /Users/name/Faces_Only
  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --confidence 0.7
  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --device cpu
  python face_detection_filter_retinaface.py -s /path/to/source -d /path/to/destination --dry-run
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
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Device to use for inference (auto, mps, cpu, default: auto)",
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

    # Map 'auto' to None for automatic detection
    device = None if args.device == "auto" else args.device

    # Create and run the face detection filter
    filter = FaceDetectionFilter(
        source_dir=args.source,
        destination_dir=args.destination,
        confidence_threshold=args.confidence,
        device=device,
        max_workers=args.workers,
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
