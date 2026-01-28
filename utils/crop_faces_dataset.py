"""Crop every face in a dataset into a dedicated output tree.

This utility walks the source dataset, detects faces with MediaPipe, expands
the crop by a configurable padding, and saves each face crop into a mirror tree
rooted under `[DATASETNAME]_cropped`. Images where no crop is produced are
copied into `[DATASETNAME]_SKIPPED` for later review.

Example:
    python utils/crop_faces_dataset.py /path/to/my_faces
    python utils/crop_faces_dataset.py ~/datasets/celeba --min-confidence 0.7
    python utils/crop_faces_dataset.py data/train --output data/train_faces --log-level DEBUG

"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import dlib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for RetinaFace
    torch = None

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/"
    "float16/1/blaze_face_short_range.tflite"
)
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

LANDMARK_DEFAULT_MODEL = Path(
    "DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
)

CACHE_DIR = Path.home() / ".crop_faces_dataset"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
DNN_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
DNN_PROTO_PATH = CACHE_DIR / "deploy.prototxt"
DNN_MODEL_PATH = CACHE_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

logger = logging.getLogger(__name__)

class DetectionBackend(str, Enum):
    MEDIAPIPE = "mediapipe"
    RETINAFACE = "retinaface"
    DNN = "dnn"


def ensure_model_downloaded() -> Path:
    """Download the MediaPipe model if it's not already cached."""
    model_dir = Path.home() / ".mediapipe" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_detection_short_range.tflite"

    if not model_path.exists():
        logger.info("Downloading MediaPipe face detection model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info("Model downloaded to %s", model_path)

    return model_path


class FaceBBoxDetector(Protocol):
    def detect_bboxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        ...


class MediaPipeFaceDetector:
    """Wraps MediaPipe face detection and produces bounding boxes."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        model_path = ensure_model_downloaded()
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_bboxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detections = self.detector.detect(mp_image)
        return [
            coords
            for coords in (
                detection_bbox_to_coords(det, image.shape) for det in detections
            )
            if coords
        ]


def load_landmark_resources(
    model_path: Path,
) -> tuple[dlib.fhog_object_detector | None, dlib.shape_predictor | None]:
    resolved = model_path.expanduser().resolve()
    if not resolved.exists():
        logger.warning("Landmark model not found at %s; skipping landmark mode.", resolved)
        return None, None

    try:
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(resolved))
        logger.info("Using landmark model at %s", resolved)
        return face_detector, predictor
    except Exception as exc:
        logger.warning("Failed to load landmark model %s: %s", resolved, exc)
        return None, None


def detect_landmark_bbox(
    image: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
) -> tuple[int, int, int, int] | None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    if not faces:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = predictor(rgb, face)
    coords = np.array([[pt.x, pt.y] for pt in shape.parts()], dtype=np.int64)
    if coords.size == 0:
        return (
            max(0, face.left()),
            max(0, face.top()),
            min(image.shape[1], face.right()),
            min(image.shape[0], face.bottom()),
        )

    xs = coords[:, 0]
    ys = coords[:, 1]
    left = max(0, min(int(xs.min()), face.left()))
    top = max(0, min(int(ys.min()), face.top()))
    right = min(image.shape[1], max(int(xs.max()), face.right()))
    bottom = min(image.shape[0], max(int(ys.max()), face.bottom()))

    return left, top, right, bottom


def detection_bbox_to_coords(
    detection: vision.FaceDetectorResult.Detection,
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int] | None:
    bbox = detection.bounding_box
    if not bbox:
        return None

    x0 = max(0, round(getattr(bbox, "origin_x", 0)))
    y0 = max(0, round(getattr(bbox, "origin_y", 0)))
    width = max(0, round(getattr(bbox, "width", 0)))
    height = max(0, round(getattr(bbox, "height", 0)))
    x1 = min(image_shape[1], x0 + max(1, width))
    y1 = min(image_shape[0], y0 + max(1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def expand_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    image_shape: tuple[int, int, int],
    padding: float,
    padding_top: float,
) -> tuple[int, int, int, int] | None:
    width = x1 - x0
    height = y1 - y0

    pad_w = round(width * padding)
    pad_h = round(height * padding)
    extra_top = round(height * padding_top)

    new_x0 = max(0, x0 - pad_w)
    new_y0 = max(0, y0 - pad_h - extra_top)
    new_x1 = min(image_shape[1], x1 + pad_w)
    new_y1 = min(image_shape[0], y1 + pad_h)

    if new_x1 <= new_x0 or new_y1 <= new_y0:
        return None

    return new_x0, new_y0, new_x1, new_y1


def _download_if_missing(url: str, destination: Path, description: str) -> None:
    if destination.exists():
        return


    logger.info("Downloading %s to %s", description, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


class RetinaFaceBBoxDetector:
    """RetinaFace via insightface for high accuracy detections."""

    def __init__(self, min_confidence: float = 0.5, device: str | None = None) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            msg = "RetinaFace detection requires insightface. Install via `pip install insightface`"
            raise RuntimeError(
                msg,
            ) from exc

        if device is None:
            if torch is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.confidence_threshold = min_confidence
        self.logger = logging.getLogger("RetinaFaceBBoxDetector")
        self.logger.info("Initializing RetinaFace detector on %s", self.device)

        providers = (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if self.device == "mps"
            else ["CPUExecutionProvider"]
        )

        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allow_modules=["detection"],
        )
        self.model.prepare(ctx_id=0, det_thresh=self.confidence_threshold)

    def detect_bboxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self.model.get(image)
        bboxes: list[tuple[int, int, int, int]] = []

        for face in faces:
            confidence = getattr(face, "det_score", None)
            if confidence is None:
                confidence = float(face[4]) if len(face) > 4 else 0.0

            if confidence < self.confidence_threshold:
                continue

            coords = getattr(face, "bbox", None)
            if coords is None:
                continue

            left, top, right, bottom = map(int, coords[:4])
            left = max(0, left)
            top = max(0, top)
            right = min(image.shape[1], right)
            bottom = min(image.shape[0], bottom)
            if right <= left or bottom <= top:
                continue

            bboxes.append((left, top, right, bottom))

        return bboxes


class OpenCvDnnFaceDetector:
    """OpenCV DNN face detector using the popular res10 SSD model."""

    def __init__(
        self,
        prototxt: Path,
        model: Path,
        min_confidence: float = 0.5,
    ) -> None:
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        self.confidence = min_confidence

    def detect_bboxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes: list[tuple[int, int, int, int]] = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence:
                continue

            left = int(detections[0, 0, i, 3] * w)
            top = int(detections[0, 0, i, 4] * h)
            right = int(detections[0, 0, i, 5] * w)
            bottom = int(detections[0, 0, i, 6] * h)

            left = max(0, min(left, w - 1))
            top = max(0, min(top, h - 1))
            right = max(0, min(right, w - 1))
            bottom = max(0, min(bottom, h - 1))

            if right <= left or bottom <= top:
                continue

            boxes.append((left, top, right, bottom))
        return boxes


def ensure_dnn_models(
    prototxt: Path | None,
    model: Path | None,
) -> tuple[Path, Path]:
    proto_path = prototxt or DNN_PROTO_PATH
    model_path = model or DNN_MODEL_PATH

    _download_if_missing(DNN_PROTO_URL, proto_path, "DNN prototxt")
    _download_if_missing(DNN_MODEL_URL, model_path, "DNN model")

    return proto_path, model_path


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logger.info),
        format="[%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces and save cropped faces under a parallel directory tree.",
    )
    parser.add_argument("dataset", help="Root path to the dataset to process.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output root directory. Defaults to `<dataset>_cropped` alongside the dataset.",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum detection confidence (0.0-1.0).",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=float,
        default=0.3,
        help="Relative padding added around each detection (fraction of width/height).",
    )
    parser.add_argument(
        "--padding-top",
        type=float,
        default=0.1,
        help="Additional padding fraction applied above the detected face only.",
    )

    parser.add_argument(
        "--detector",
        type=str,
        choices=[backend.value for backend in DetectionBackend],
        default=DetectionBackend.MEDIAPIPE.value,
        help="Face detection backend to use when landmarks are not requested.",
    )
    parser.add_argument(
        "--retinaface-device",
        type=str,
        choices=["cpu", "mps"],
        default=None,
        help="Preferred device for InsightFace RetinaFace (auto if unset).",
    )
    parser.add_argument(
        "--dnn-prototxt",
        type=Path,
        default=None,
        help="Custom path for the OpenCV DNN deploy prototxt.",
    )
    parser.add_argument(
        "--dnn-model",
        type=Path,
        default=None,
        help="Custom path for the OpenCV DNN caffemodel weights.",
    )
    parser.add_argument(
        "--use-landmarks",
        action="store_true",
        help="Use the dlib shape predictor to locate faces before cropping.",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=LANDMARK_DEFAULT_MODEL,
        help="Path to the dlib shape predictor model file.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def build_backend_detector(args: argparse.Namespace) -> FaceBBoxDetector:
    backend = DetectionBackend(args.detector)
    if backend == DetectionBackend.MEDIAPIPE:
        return MediaPipeFaceDetector(min_confidence=args.min_confidence)
    if backend == DetectionBackend.RETINAFACE:
        return RetinaFaceBBoxDetector(
            min_confidence=args.min_confidence,
            device=args.retinaface_device,
        )

    proto_path, model_path = ensure_dnn_models(args.dnn_prototxt, args.dnn_model)
    logger.info("Using OpenCV DNN model at %s", model_path)
    return OpenCvDnnFaceDetector(proto_path, model_path, min_confidence=args.min_confidence)


def gather_images(dataset_root: Path, exclude_roots: Iterable[Path]) -> list[Path]:
    """Return sorted image paths while skipping excluded subtrees."""
    resolved_dataset = dataset_root.resolve()
    excluded = [root.resolve() for root in exclude_roots]
    image_files: list[Path] = []

    for path in resolved_dataset.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(_is_under(path, root) for root in excluded):
            continue
        image_files.append(path)

    image_files.sort()
    return image_files


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def copy_to_skipped(image_path: Path, dataset_root: Path, skipped_root: Path) -> None:
    try:
        relative_dir = image_path.parent.relative_to(dataset_root)
        target_dir = skipped_root / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_dir / image_path.name)
    except Exception as exc:
        logger.warning("Failed to copy skipped image %s: %s", image_path, exc)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        logger.error("Dataset path must exist and be a directory: %s", dataset_root)
        sys.exit(1)

    dataset_name = dataset_root.name
    if not dataset_name or dataset_name == ".":
        dataset_name = dataset_root.resolve().name or "dataset"

    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        output_root = dataset_root.parent / f"{dataset_name}_cropped"

    if output_root == dataset_root:
        logger.error("Output root cannot be the same as the dataset root.")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    skipped_root = dataset_root.parent / f"{dataset_name}_SKIPPED"
    skipped_root.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset root: %s", dataset_root)
    logger.info("Output root: %s", output_root)
    logger.info("Skipped root: %s", skipped_root)
    logger.info("Minimum confidence: %.2f", args.min_confidence)
    logger.info("Padding fraction: %.2f", args.padding)
    logger.info("Top padding fraction: %.2f", args.padding_top)
    logger.info("Landmark mode: %s", args.use_landmarks)
    logger.info("Detector backend: %s", args.detector)
    if args.retinaface_device:
        logger.info("RetinaFace device hint: %s", args.retinaface_device)

    if args.padding < 0 or args.padding_top < 0:
        logger.error("Padding must be non-negative.")
        sys.exit(1)

    landmark_detector = None
    landmark_predictor = None
    if args.use_landmarks:
        detector, predictor = load_landmark_resources(args.landmark_model)
        if detector is not None and predictor is not None:
            landmark_detector = detector
            landmark_predictor = predictor
        else:
            logger.warning("Landmark mode disabled, falling back to MediaPipe detection.")

    image_paths = gather_images(dataset_root, [output_root, skipped_root])
    if not image_paths:
        logger.warning("No images found under %s", dataset_root)
        return

    backend_detector = build_backend_detector(args)
    stats = {
        "processed": 0,
        "images_with_faces": 0,
        "faces_saved": 0,
        "errors": 0,
        "skipped": 0,
    }

    for image_path in tqdm(image_paths, desc="Processing", unit="image"):
        stats["processed"] += 1
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Unable to read image: %s", image_path)
                stats["errors"] += 1
                continue

            bbox_coords = None
            if landmark_detector and landmark_predictor:
                bbox_coords = detect_landmark_bbox(
                    image,
                    landmark_detector,
                    landmark_predictor,
                )

            if bbox_coords is None:
                detections = backend_detector.detect_bboxes(image)
                if not detections:
                    stats["skipped"] += 1
                    copy_to_skipped(image_path, dataset_root, skipped_root)
                    continue

                bbox_coords = detections[0]

            expanded = expand_bbox(
                *bbox_coords,
                image.shape,
                args.padding,
                args.padding_top,
            )
            if expanded is None:
                stats["skipped"] += 1
                copy_to_skipped(image_path, dataset_root, skipped_root)
                continue

            x0, y0, x1, y1 = expanded
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                stats["skipped"] += 1
                copy_to_skipped(image_path, dataset_root, skipped_root)
                continue

            stats["images_with_faces"] += 1
            rel_dir = image_path.parent.relative_to(dataset_root)
            target_dir = output_root / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"{image_path.stem}_face01.jpg"
            dest_path = target_dir / dest_name
            cv2.imwrite(str(dest_path), crop)
            stats["faces_saved"] += 1
        except Exception as exc:  # pragma: no cover - best effort batch process
            logger.exception("Failed processing %s: %s", image_path, exc)
            stats["errors"] += 1

    logger.info("Processed %d images", stats["processed"])
    logger.info("Images with faces: %d", stats["images_with_faces"])
    logger.info("Faces saved: %d", stats["faces_saved"])
    logger.info("Images skipped (no faces or failed crops): %d", stats["skipped"])
    logger.info("Errors: %d", stats["errors"])


if __name__ == "__main__":
    main()
