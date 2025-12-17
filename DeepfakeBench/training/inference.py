"""Refactored deepfake detection inference script.

Follows context7 and ruff best practices for maintainable code.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import IO, Any

import cv2
import dlib
import numpy as np
import torch
import yaml
from detectors import DETECTOR
from imutils import face_utils
from PIL import Image
from skimage import transform as trans
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

# Configuration constants
DEFAULT_RESOLUTION = 224
DEFAULT_TARGET_SIZE = (112, 112)
TARGET_SIZE_112 = 112
DEFAULT_SCALE = 1.3
MIN_CLASSES_FOR_AUC = 2

# Landmark indices for face alignment
LANDMARK_INDICES = {
    "leye": 37,
    "reye": 44,
    "nose": 30,
    "lmouth": 49,
    "rmouth": 55,
}

# Label constants
REAL_LABEL = 0
FAKE_LABEL = 1

# Keyword synonyms for flexible directory matching
REAL_KEYWORDS = ["real", "original", "authentic", "genuine", "nature", "natural"]
FAKE_KEYWORDS = [
    "fake",
    "synthetic",
    "ai",
    "artificial",
    "manipulated",
    "deepfake",
    "faceswap",
    "synthesis",
]

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages PyTorch device selection following context7 best practices."""

    @staticmethod
    def get_optimal_device() -> torch.device:
        """Get optimal device with priority: CUDA > MPS > CPU."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA (NVIDIA GPU)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device


class FaceKeypointExtractor:
    """Extracts facial keypoints using dlib landmarks."""

    @staticmethod
    def get_keypts(
        image: np.ndarray,
        face: dlib.rectangle,
        predictor: dlib.shape_predictor,
    ) -> np.ndarray:
        """Extract key facial points for alignment."""
        shape = predictor(image, face)

        # Extract key points
        points = []
        # for _name, idx in LANDMARK_INDICES.items():
        for idx in LANDMARK_INDICES.values():
            point = np.array([shape.part(idx).x, shape.part(idx).y]).reshape(-1, 2)
            points.append(point)

        return np.concatenate(points, axis=0)


class FaceAlignment:
    """Handles face detection and alignment operations."""

    def __init__(
        self,
        face_detector: dlib.fhog_object_detector | None = None,
        predictor: dlib.shape_predictor | None = None,
    ) -> None:
        """Initialize face alignment with detectors."""
        self.face_detector = face_detector
        self.predictor = predictor

    def extract_aligned_face(
        self,
        image: np.ndarray,
        res: int = DEFAULT_RESOLUTION,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dlib.rectangle | None]:
        """Extract and align face from image with fallback handling."""
        image.shape[:2]  # Get height and width but don't store

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Try face detection if detectors available
        if self.face_detector is None or self.predictor is None:
            return self._fallback_no_detection(rgb, res, mask)

        faces = self.face_detector(rgb, 1)
        if not faces:
            return self._fallback_no_detection(rgb, res, mask)

        # Use the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Get landmarks
        landmarks = FaceKeypointExtractor.get_keypts(rgb, face, self.predictor)

        # Align and crop
        cropped_face = self._align_and_crop(
            rgb,
            landmarks,
            outsize=(res, res),
            scale=DEFAULT_SCALE,
            mask=mask,
        )
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Extract landmarks from aligned face
        face_align = self.face_detector(cropped_face, 1)
        landmark = None

        if len(face_align) > 0:
            landmark_shape = self.predictor(cropped_face, face_align[0])
            landmark = face_utils.shape_to_np(landmark_shape)

        return cropped_face, landmark, face

    def _align_and_crop(
        self,
        rgb: np.ndarray,
        landmarks: np.ndarray,
        outsize: tuple[int, int],
        scale: float,
        mask: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Align and crop face based on landmarks."""
        transform_matrix = self._compute_transform_matrix(landmarks, outsize, scale)

        target_height, target_width = outsize
        transformed = cv2.warpAffine(
            rgb,
            transform_matrix,
            (target_width, target_height),
        )

        if outsize != (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0]):
            transformed = cv2.resize(transformed, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, transform_matrix, (target_width, target_height))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return transformed, mask

        return transformed

    def _compute_transform_matrix(
        self,
        landmarks: np.ndarray,
        outsize: tuple[int, int],
        scale: float,
    ) -> np.ndarray:
        """Compute affine transformation matrix for face alignment."""
        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        # Adjust for 112x112 target
        if outsize[1] == TARGET_SIZE_112:
            dst[:, 0] += 8.0

        # Scale destination points
        dst[:, 0] *= outsize[0] / DEFAULT_TARGET_SIZE[0]
        dst[:, 1] *= outsize[1] / DEFAULT_TARGET_SIZE[1]

        # Apply scaling and centering
        target_size = outsize
        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        # Compute transformation
        src = landmarks.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        return tform.params[0:2, :]

    def _fallback_no_detection(
        self,
        _rgb: np.ndarray,
        _res: int,
        _mask: np.ndarray | None = None,
    ) -> tuple[None, None, None]:
        """Fallback when no face detector is available."""
        return None, None, None


class ModelLoader:
    """Handles detector model loading with proper error handling."""

    @staticmethod
    def load_detector(
        detector_cfg: str,
        weights: str,
        device: torch.device,
    ) -> torch.nn.Module:
        """Load and configure detector model."""
        config_path = Path(detector_cfg)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        model_cls = DETECTOR[cfg["model_name"]]
        model = model_cls(cfg).to(device)

        ckpt = torch.load(weights, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()

        logger.info("Detector loaded successfully")
        return model


class ImagePreprocessor:
    """Handles image preprocessing for model inference."""

    @staticmethod
    def preprocess_face(img_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR image to model-ready tensor."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb,
            (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
            interpolation=cv2.INTER_LINEAR,
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ],
        )
        return transform(Image.fromarray(img_rgb)).unsqueeze(0)


class InferenceEngine:
    """Handles model inference with device context management."""

    @staticmethod
    @torch.inference_mode()
    def infer_single_image(
        img_bgr: np.ndarray,
        face_detector: FaceAlignment | None,
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[int, float]:
        """Perform inference on single image with proper device handling."""
        if face_detector is None:
            face_aligned = img_bgr
        else:
            face_result = face_detector.extract_aligned_face(
                img_bgr,
                res=DEFAULT_RESOLUTION,
            )
            face_aligned = face_result[0] if face_result[0] is not None else img_bgr

        face_tensor = ImagePreprocessor.preprocess_face(face_aligned).to(device)
        data = {"image": face_tensor, "label": torch.tensor([REAL_LABEL]).to(device)}

        with torch.device(device):
            predictions = InferenceEngine._run_model(model, data)

        cls_out = predictions["cls"].squeeze().cpu().numpy()
        prob = predictions["prob"].squeeze().cpu().numpy()

        # Handle numpy array conversion
        if isinstance(cls_out, np.ndarray):
            cls_out = cls_out.item() if cls_out.ndim == 0 else int(np.argmax(cls_out))
        if isinstance(prob, np.ndarray):
            prob = prob.item() if prob.ndim == 0 else float(prob)

        return cls_out, prob

    @staticmethod
    @torch.no_grad()
    def _run_model(
        model: torch.nn.Module,
        data_dict: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run model inference with gradient disabled."""
        return model(data_dict, inference=True)  # type: ignore[return-value]


class ImageCollection:
    """Handles image path collection and validation."""

    @staticmethod
    def collect_image_paths(path_str: str, limit: int = 1000) -> list[Path]:
        """Collect valid image paths from given directory or file."""
        p = Path(path_str)
        ImageCollection._validate_path(p, path_str)

        if p.is_file():
            return [p]

        img_list = ImageCollection._find_images_in_directory(p)
        ImageCollection._validate_image_count(img_list, path_str)

        return ImageCollection._sample_images(img_list, limit)

    @staticmethod
    def _validate_path(p: Path, path_str: str) -> None:
        """Validate input path exists."""
        if not p.exists():
            error_msg = f"Path does not exist: {path_str}"
            raise FileNotFoundError(error_msg)

    @staticmethod
    def _find_images_in_directory(p: Path) -> list[Path]:
        """Find images in directory (first level, then recursive)."""
        # Try root directory first
        img_list = [
            fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS
        ]

        # If none found, search recursively
        if not img_list:
            logger.debug("No images found in root directory, searching subdirectories")
            img_list = [
                fp
                for fp in p.rglob("*")
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS
            ]
            logger.debug("Found %d images in subdirectories", len(img_list))

        return img_list

    @staticmethod
    def _validate_image_count(img_list: list[Path], path_str: str) -> None:
        """Validate that images were found."""
        if not img_list:
            error_msg = f"No valid image files found in directory: {path_str}"
            raise RuntimeError(error_msg)

    @staticmethod
    def _sample_images(img_list: list[Path], limit: int) -> list[Path]:
        """Sample images if limit exceeded, otherwise shuffle."""
        if len(img_list) > limit:
            logger.info(
                "Randomly sampling %d images from %d available images",
                limit,
                len(img_list),
            )
            return random.sample(img_list, limit)

        # Shuffle for random order
        random.shuffle(img_list)
        return img_list


class LabelExtractor:
    """Extracts labels from directory structures."""

    @staticmethod
    def extract_labels(img_paths: list[Path], base_path: str) -> list[int]:
        """Extract true labels from directory-based naming patterns."""
        base_path_obj = Path(base_path)

        logger.debug("Analyzing %d images for label extraction", len(img_paths))
        logger.debug("Base path: %s", base_path)

        real_dirs, fake_dirs = LabelExtractor._find_label_directories(base_path_obj)
        if not (real_dirs or fake_dirs):
            logger.debug("No 'real' or 'fake' directories found")
            return []

        real_names = {d.name.lower() for d in real_dirs}
        fake_names = {d.name.lower() for d in fake_dirs}

        return LabelExtractor._extract_labels_from_paths(
            img_paths,
            real_names,
            fake_names,
        )

    @staticmethod
    def _find_label_directories(base_path: Path) -> tuple[list[Path], list[Path]]:
        """Find directories containing real/fake labels."""
        if not base_path.exists():
            logger.debug("Base path does not exist: %s", base_path)
            return [], []

        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        logger.debug(
            "Found %d subdirectories: %s",
            len(subdirs),
            [d.name for d in subdirs],
        )

        real_dirs = [
            d for d in subdirs if any(kw in d.name.lower() for kw in REAL_KEYWORDS)
        ]
        fake_dirs = [
            d for d in subdirs if any(kw in d.name.lower() for kw in FAKE_KEYWORDS)
        ]

        logger.debug("Real directories found: %s", [d.name for d in real_dirs])
        logger.debug("Fake directories found: %s", [d.name for d in fake_dirs])

        return real_dirs, fake_dirs

    @staticmethod
    def _extract_labels_from_paths(
        img_paths: list[Path],
        real_names: set,
        fake_names: set,
    ) -> list[int]:
        """Extract labels by checking directory names at multiple levels."""
        labels = []

        for img_path in img_paths:
            label = LabelExtractor._find_label_for_path(
                img_path,
                real_names,
                fake_names,
            )
            if label is not None:
                labels.append(label)

        return labels

    @staticmethod
    def _find_label_for_path(
        img_path: Path,
        real_names: set,
        fake_names: set,
    ) -> int | None:
        """Find label for image by checking directory names."""
        # Check directory names at different levels
        dir_names = []
        for i in range(1, 6):  # Check up to 5 levels
            if len(img_path.parts) >= i:
                dir_names.append(img_path.parts[-i].lower())
            else:
                dir_names.append("")

        for dir_name in dir_names:
            if dir_name in real_names:
                return REAL_LABEL
            if dir_name in fake_names:
                return FAKE_LABEL

        return None


class PerformanceCalculator:
    """Calculate performance metrics from predictions and true labels."""

    @staticmethod
    def calculate_metrics(
        true_labels: list[int],
        predictions: list[int],
        probabilities: list[float],
    ) -> dict[str, Any]:
        """Calculate AUC and PR-AUC metrics."""
        if len(set(true_labels)) < MIN_CLASSES_FOR_AUC:
            return {"error": "Only one class present - cannot calculate AUC"}

        try:
            auc_score = roc_auc_score(true_labels, probabilities)
            pr_auc_score = average_precision_score(true_labels, probabilities)

            return {
                "auc": auc_score,
                "pr_auc": pr_auc_score,
                "total_images": len(predictions),
                "real_count": sum(1 for label in true_labels if label == REAL_LABEL),
                "fake_count": sum(1 for label in true_labels if label == FAKE_LABEL),
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"Error calculating metrics: {e}"}


class ResultWriter:
    """Handles writing inference results to files."""

    def __init__(self, filename: str = "inference_results.txt") -> None:
        """Initialize result writer with output filename."""
        self.filename = filename

    def write_results(
        self,
        results: list[dict[str, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Write all results to output file."""
        output_path = Path(self.filename)
        with output_path.open("w") as f:
            self._write_header(f)

            for result in results:
                self._write_single_result(f, result)

            if metrics:
                self._write_metrics_section(f, metrics)

    def _write_header(self, f: IO) -> None:
        """Write file header."""
        f.write("Deepfake Detection Results\n")
        f.write("=" * 80 + "\n\n")

    def _write_single_result(self, f: IO[str], result: dict[str, Any]) -> None:
        """Write single result line."""
        f.write(
            f"[{result['index']}/{result['total']}] {result['filename']:>30} | "
            f"True: {result['true_label']:>4} | Pred: {result['prediction']} "
            f"(0=Real, 1=Fake) | Prob: {result['probability']:.4f} | "
            f"Path: {result['path']}\n",
        )

    def _write_metrics_section(self, f: IO[str], metrics: dict[str, Any]) -> None:
        """Write metrics section."""
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n")

        if "error" in metrics:
            f.write(f"\n[WARNING] {metrics['error']}\n")
        else:
            self._write_detailed_metrics(f, metrics)

    def _write_detailed_metrics(self, f: IO[str], metrics: dict[str, Any]) -> None:
        """Write detailed metrics information."""
        f.write(f"\nTotal images processed: {metrics['total_images']}\n")
        f.write(f"Real images: {metrics['real_count']}\n")
        f.write(f"Fake images: {metrics['fake_count']}\n")
        f.write(f"AUC (Area Under ROC Curve): {metrics['auc']:.4f}\n")
        f.write(
            f"PR-AUC (Area Under Precision-Recall Curve): {metrics['pr_auc']:.4f}\n",
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deepfake image inference (refactored version)",
    )
    parser.add_argument(
        "--detector_config",
        default="training/config/detector/effort.yaml",
        help="YAML configuration file path",
    )
    parser.add_argument("--weights", required=True, help="Detector pretrained weights")
    parser.add_argument("--image", required=True, help="Tested image or directory")
    parser.add_argument(
        "--landmark_model",
        default=False,
        help="dlib 81 landmarks dat file / False if no face cropping needed",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the main inference workflow."""
    args = parse_args()

    # Initialize device
    device = DeviceManager.get_optimal_device()

    # Load model
    model = ModelLoader.load_detector(args.detector_config, args.weights, device)

    # Initialize face detector if landmark model provided
    face_detector = None
    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(args.landmark_model)
        face_detector = FaceAlignment(face_det, shape_predictor)

    # Collect images
    img_paths = ImageCollection.collect_image_paths(args.image)
    is_multiple = len(img_paths) > 1

    if is_multiple:
        logger.info("Collected %d images in total, let's infer them...", len(img_paths))

        # Extract true labels for multiple image analysis
        true_labels = LabelExtractor.extract_labels(img_paths, args.image)
    else:
        true_labels = []

    # Perform inference
    results = []
    predictions = []
    probabilities = []

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to load image, skipping: %s", img_path)
            continue

        cls, prob = InferenceEngine.infer_single_image(
            img,
            face_detector,
            model,
            device,
        )

        # Store results
        predictions.append(int(cls))
        probabilities.append(float(prob))

        # Find corresponding true label
        true_label = "N/A"
        if is_multiple and idx <= len(true_labels):
            true_label_str = "REAL" if true_labels[idx - 1] == REAL_LABEL else "FAKE"
            true_label = true_label_str

        results.append(
            {
                "index": idx,
                "total": len(img_paths),
                "filename": img_path.name,
                "true_label": true_label,
                "prediction": cls,
                "probability": prob,
                "path": img_path,
            },
        )

    # Calculate metrics
    metrics = None
    if len(predictions) > 1 and true_labels:
        valid_true_labels = [label for label in true_labels if label is not None]
        if valid_true_labels:
            metrics = PerformanceCalculator.calculate_metrics(
                valid_true_labels,
                predictions,
                probabilities,
            )

            # Print metrics to console
            if "auc" in metrics:
                logger.info("AUC (Area Under ROC Curve): %.4f", metrics["auc"])
                logger.info(
                    "PR-AUC (Area Under Precision-Recall Curve): %.4f",
                    metrics["pr_auc"],
                )

    # Write results
    writer = ResultWriter()
    writer.write_results(results, metrics)
    logger.info("Results written to %s", writer.filename)


if __name__ == "__main__":
    main()
