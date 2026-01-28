"""Deepfake Detection Server.

This module provides a FastAPI server for deepfake detection using the Effort model.
It includes endpoints for health checks and image prediction with Grad-CAM visualization.
"""

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

# Add DeepfakeBench and backend to sys.path to allow imports
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(current_dir))

import cv2
import dlib
import mediapipe as mp
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradcam_utils import reshape_transform
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydantic import BaseModel

# Import Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from video_utils import extract_frames

from DeepfakeBench.training.detectors import DETECTOR
from DeepfakeBench.training.inference import (
    DeviceManager,
    FaceAlignment,
    ImagePreprocessor,
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    """Wrapper to make EffortDetector compatible with Grad-CAM."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize ModelWrapper.

        Args:
            model: The model to wrap for Grad-CAM compatibility.

        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Grad-CAM.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from the model.

        """
        return self.model({"image": x})["cls"]


# Configuration constants
class Config:
    """Configuration constants for the server."""

    CONFIG_PATH = Path(
        "DeepfakeBench/training/config/detector/effort_finetune.yaml",
    )
    WEIGHTS_PATH = Path(
        "DeepfakeBench/training/weights/finetuned/newBatchFaces.pth",
    )
    LANDMARK_MODEL_PATH = Path(
        "DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
        # "DeepfakeBench/preprocessing/missing.dat",
    )

    # Landmark group indices for 81-landmark model
    LANDMARK_GROUPS: ClassVar[dict[str, list]] = {
        "eyes": list(range(36, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
        "forehead": list(range(68, 81)),
    }

    # Threshold constants
    SUBTLE_FEATURES_THRESHOLD: ClassVar[float] = 0.2
    FAKE_PROBABILITY_THRESHOLD: ClassVar[float] = 0.5

    # MediaPipe face detection constants
    USE_MEDIAPIPE: ClassVar[bool] = os.environ.get("SERVER_USE_MEDIAPIPE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    MEDIAPIPE_MIN_CONFIDENCE: ClassVar[float] = float(
        os.environ.get("SERVER_MEDIAPIPE_CONFIDENCE", "0.5"),
    )
    MEDIAPIPE_PADDING: ClassVar[float] = float(
        os.environ.get("SERVER_MEDIAPIPE_PADDING", "0.2"),
    )


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/"
    "float16/1/blaze_face_short_range.tflite"
)


def _get_mediapipe_model_path() -> Path:
    model_dir = Path.home() / ".mediapipe" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_detection_short_range.tflite"
    if not model_path.exists():
        logger.info("Downloading MediaPipe face detection model to %s", model_path)
        import urllib.request

        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class ServerMediaPipeFaceDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        model_path = _get_mediapipe_model_path()
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[vision.FaceDetectorResult.Detection]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.detector.detect(mp_image)
        return list(result.detections)


def detection_bbox_to_coords(
    detection: vision.FaceDetectorResult.Detection,
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int] | None:
    bbox = detection.bounding_box
    if not bbox:
        return None

    x0 = max(0, int(round(getattr(bbox, "origin_x", 0))))
    y0 = max(0, int(round(getattr(bbox, "origin_y", 0))))
    width = max(0, int(round(getattr(bbox, "width", 0))))
    height = max(0, int(round(getattr(bbox, "height", 0))))
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
) -> tuple[int, int, int, int] | None:
    width = x1 - x0
    height = y1 - y0

    pad_w = int(round(width * padding))
    pad_h = int(round(height * padding))
    new_x0 = max(0, x0 - pad_w)
    new_y0 = max(0, y0 - pad_h)
    new_x1 = min(image_shape[1], x1 + pad_w)
    new_y1 = min(image_shape[0], y1 + pad_h)

    if new_x1 <= new_x0 or new_y1 <= new_y0:
        return None

    return new_x0, new_y0, new_x1, new_y1


class ServerState:
    """Global server state management."""

    def __init__(self) -> None:
        """Initialize server state."""
        self.device = None
        self.model = None
        self.cam = None
        self.face_detector = None
        self.mediapipe_detector = None

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "cam") and self.cam is not None:
            del self.cam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.mediapipe_detector = None


# Global server state
server_state = ServerState()


def find_best_weights() -> Path | None:
    """Find the best available model weights.

    Returns:
        Path to the best weights file, or None if not found.

    """
    # Use the hardcoded path requested by the user
    if Config.WEIGHTS_PATH.exists():
        logger.info("Found weights at requested path: %s", Config.WEIGHTS_PATH)
        return Config.WEIGHTS_PATH

    logger.warning(
        "Requested weights not found at %s. Falling back to search.",
        Config.WEIGHTS_PATH,
    )

    # Attempt to find the best weights in the logs directory
    # Only looking at 'effort' related folders
    logs_dir = Path("DeepfakeBench/training/logs")
    if not logs_dir.exists():
        return None

    # Simple heuristic: find most recent modified weights (*.pth) in subdirs
    candidates = [path for path in logs_dir.rglob("*.pth") if path.is_file()]
    if not candidates:
        return None

    # Sort by modification time, newest first
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    logger.info("Found weights: %s", [str(c) for c in candidates])
    return candidates[0]


def load_model_config() -> dict[str, Any]:
    """Load model configuration from YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        RuntimeError: If config file is not found.

    """
    if not Config.CONFIG_PATH.exists():
        config_error_msg = f"Config not found at {Config.CONFIG_PATH}"
        logger.error(config_error_msg)
        raise RuntimeError("Config file not found")

    with Config.CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def load_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load and initialize the deepfake detection model.

    Args:
        config: Model configuration dictionary.
        device: Target device for the model.

    Returns:
        Initialized model.

    Raises:
        RuntimeError: If model definition is not found.

    """
    try:
        return DETECTOR[config["model_name"]](config).to(device)
    except KeyError as e:
        logger.exception(
            "Model %s not found in DETECTOR registry",
            config["model_name"],
        )
        model_error_msg = "Model definition not found"
        raise RuntimeError(model_error_msg) from e


def load_model_weights(model: torch.nn.Module, weights_path: Path) -> None:
    """Load model weights from checkpoint file.

    Args:
        model: Model to load weights into.
        weights_path: Path to weights file.

    """
    logger.info("Loading weights from %s", weights_path)
    ckpt = torch.load(weights_path, map_location=server_state.device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Remove module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)


def setup_grad_cam(model: torch.nn.Module) -> GradCAM:
    """Set up Grad-CAM for model visualization.

    Args:
        model: Model to set up Grad-CAM for.

    Returns:
        Configured Grad-CAM instance.

    """
    # Target layer for ViT backbone in Effort model
    # Backbone is clip_model.vision_model
    # We target the last layer norm of the encoder
    target_layers = [model.backbone.encoder.layers[-1].layer_norm1]

    # Wrap model for Grad-CAM
    wrapped_model = ModelWrapper(model)

    return GradCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )


def load_mediapipe_detector() -> ServerMediaPipeFaceDetector | None:
    """Load MediaPipe face detector when enabled."""
    if not Config.USE_MEDIAPIPE:
        return None

    try:
        detector = ServerMediaPipeFaceDetector(Config.MEDIAPIPE_MIN_CONFIDENCE)
        logger.info("MediaPipe face detector enabled")
        return detector
    except Exception as exc:
        logger.warning("Failed to initialize MediaPipe detector: %s", exc)
        return None


def load_face_detector() -> FaceAlignment | None:
    """Load face detector for face alignment.

    Returns:
        FaceAlignment instance if landmark model is found, None otherwise.

    """
    if Config.LANDMARK_MODEL_PATH.exists():
        logger.info("Loading landmark model from %s", Config.LANDMARK_MODEL_PATH)
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(str(Config.LANDMARK_MODEL_PATH))
        return FaceAlignment(face_det, shape_predictor)
    logger.warning(
        "Landmark model not found at %s. Face alignment disabled.",
        Config.LANDMARK_MODEL_PATH,
    )
    return None


def _crop_with_mediapipe(image: np.ndarray) -> np.ndarray | None:
    detector = server_state.mediapipe_detector
    if detector is None:
        return None

    detections = detector.detect(image)
    if not detections:
        return None

    bbox_coords = detection_bbox_to_coords(detections[0], image.shape)
    if bbox_coords is None:
        return None

    expanded = expand_bbox(*bbox_coords, image.shape, Config.MEDIAPIPE_PADDING)
    if expanded is None:
        return None

    x0, y0, x1, y1 = expanded
    return image[y0:y1, x0:x1]


def align_face(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    face_aligned = image
    landmarks = None

    if server_state.face_detector:
        face_result = server_state.face_detector.extract_aligned_face(image, res=224)
        if face_result[0] is not None:
            face_aligned = face_result[0]
            landmarks = face_result[1]
            logger.info("Face aligned successfully")
        else:
            logger.warning("No face detected by landmark model, checking fallback.")
            mp_crop = _crop_with_mediapipe(image)
            if mp_crop is not None:
                face_aligned = mp_crop
    else:
        mp_crop = _crop_with_mediapipe(image)
        if mp_crop is not None:
            face_aligned = mp_crop

    return face_aligned, landmarks


def analyze_heatmap_regions(mask: np.ndarray, landmarks: np.ndarray) -> str:
    """Analyze which facial regions are most affected by the heatmap.

    Args:
        mask: Grad-CAM heatmap mask.
        landmarks: Facial landmarks array.

    Returns:
        String describing the analysis result.

    """
    region_scores = {}

    for name, indices in Config.LANDMARK_GROUPS.items():
        # Filter indices to ensure they exist in the landmark set
        valid_indices = [idx for idx in indices if idx < len(landmarks)]
        if not valid_indices:
            continue

        region_pts = landmarks[valid_indices]

        # Create a mask for this region
        # We use a convex hull or a simple bounding box
        rect = cv2.boundingRect(region_pts.astype(np.int32))
        x, y, w, h = rect

        # Add padding to the region
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(mask.shape[1] - x, w + 2 * pad)
        h = min(mask.shape[0] - y, h + 2 * pad)

        region_mask_crop = mask[y : y + h, x : x + w]
        if region_mask_crop.size > 0:
            region_scores[name] = np.mean(region_mask_crop)

    if not region_scores:
        return "The model analyzed the overall facial structure."

    # Find the top scoring region
    top_region = max(region_scores, key=region_scores.get)
    intensity = region_scores[top_region]

    if intensity < Config.SUBTLE_FEATURES_THRESHOLD:
        return "The model's decision is based on subtle features across the face."

    descriptions = {
        "eyes": "Suspicious textures or abnormal reflections were detected around the eyes.",
        "nose": "Artifacts were detected in the nasal region and central face.",
        "mouth": "The model flagged unnatural blending or motion artifacts around the mouth.",
        "forehead": "Smoothing or skin texture inconsistencies were found on the forehead.",
    }

    return descriptions.get(
        top_region,
        f"The model is focused on the {top_region} region.",
    )


def preprocess_image(image_data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Preprocess uploaded image for model inference.

    Args:
        image_data: Raw image data bytes.

    Returns:
        Tuple of (processed_image, landmarks) where landmarks may be None.

    Raises:
        HTTPException: If image is invalid or processing fails.

    """
    # Read Image
    nparr = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face_aligned, landmarks = align_face(img_bgr)

    # Convert to RGB and resize for Grad-CAM visualization
    img_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
    img_float_norm = cv2.resize(img_rgb, (224, 224))
    img_float_norm = np.float32(img_float_norm) / 255.0

    return img_float_norm, landmarks


def perform_inference(input_tensor: torch.Tensor) -> tuple[str, float, str | None]:
    """Perform model inference and generate prediction.

    Args:
        input_tensor: Preprocessed input tensor.

    Returns:
        Tuple of (label, probability, reasoning).

    """
    # Inference
    with torch.no_grad():
        pred_dict = server_state.model(
            {"image": input_tensor, "label": torch.tensor([0]).to(server_state.device)},
        )  # Mock label
        prob = pred_dict["prob"].item()
        # prob is probability of FAKE (1).

    label_str = "FAKE" if prob > Config.FAKE_PROBABILITY_THRESHOLD else "REAL"
    reasoning = None

    return label_str, prob, reasoning


def generate_grad_cam(input_tensor: torch.Tensor, img_float_norm: np.ndarray) -> str:
    """Generate Grad-CAM visualization.

    Args:
        input_tensor: Input tensor for Grad-CAM.
        img_float_norm: Normalized image for overlay.

    Returns:
        Base64 encoded Grad-CAM image.

    """
    # Generate Grad-CAM
    grayscale_cam = server_state.cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay
    cam_image = show_cam_on_image(img_float_norm, grayscale_cam, use_rgb=True)

    # Encode Grad-CAM to Base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    grad_cam_b64 = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/jpeg;base64,{grad_cam_b64}"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> None:
    """FastAPI lifespan context manager for resource management."""
    # Initialize device
    server_state.device = DeviceManager.get_optimal_device()
    logger.info("Using device: %s", server_state.device)

    # Load configuration
    config = load_model_config()

    # Load model
    server_state.model = load_model(config, server_state.device)

    # Load weights
    weights_path = find_best_weights()
    if weights_path:
        load_model_weights(server_state.model, weights_path)
    else:
        logger.warning(
            "No checkpoint found! Using random initialization (Expect poor results)",
        )

    server_state.model.eval()

    # Setup Grad-CAM
    server_state.cam = setup_grad_cam(server_state.model)

    # Load Face Detector
    server_state.face_detector = load_face_detector()

    # Load MediaPipe detector if configured
    server_state.mediapipe_detector = load_mediapipe_detector()

    yield

    # Clean up resources
    server_state.cleanup()


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    """Response model for prediction results."""

    label: str
    score: float
    reasoning: str | None = None
    grad_cam_image: str  # Base64 encoded image


class VideoPredictionResponse(BaseModel):
    """Response model for video prediction results."""

    label: str
    score: float
    reasoning: str | None = None
    grad_cam_image: str  # Base64 encoded image from the most suspicious frame
    sampled_frames: int
    worst_frame_index: int
    worst_frame_score: float


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict deepfake probability for uploaded image.

    Args:
        file: Uploaded image file.

    Returns:
        Prediction response with label, score, reasoning, and Grad-CAM visualization.

    Raises:
        HTTPException: If model is not loaded or image processing fails.

    """
    if not server_state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read and preprocess image
    contents = await file.read()
    img_float_norm, landmarks = preprocess_image(contents)

    # Prepare tensor for model input
    input_tensor = ImagePreprocessor.preprocess_face(
        cv2.cvtColor(np.uint8(img_float_norm * 255), cv2.COLOR_RGB2BGR),
    ).to(server_state.device)

    # Perform inference
    label_str, prob, reasoning = perform_inference(input_tensor)

    # Generate reasoning based on Grad-CAM analysis
    if label_str == "FAKE" and landmarks is not None:
        grayscale_cam = server_state.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        reasoning = analyze_heatmap_regions(grayscale_cam, landmarks)

    # Generate Grad-CAM visualization
    grad_cam_image = generate_grad_cam(input_tensor, img_float_norm)

    return {
        "label": label_str,
        "score": float(prob),
        "reasoning": reasoning,
        "grad_cam_image": grad_cam_image,
    }


@app.post("/predict_video", response_model=VideoPredictionResponse)
async def predict_video(file: UploadFile = File(...)) -> VideoPredictionResponse:
    """Predict deepfake probability for an uploaded video.

    The video is sampled into a fixed number of frames, each frame is run through the
    same image pipeline, and the video-level score is the average frame score.

    Returns Grad-CAM + reasoning for the most suspicious (highest fake-probability)
    sampled frame.
    """
    if not server_state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()

    try:
        extracted = extract_frames(contents, num_frames=60)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    frame_scores: list[float] = []
    worst_idx = -1
    worst_score = -1.0
    worst_input_tensor: torch.Tensor | None = None
    worst_img_float_norm: np.ndarray | None = None
    worst_landmarks: np.ndarray | None = None

    for frame in extracted:
        # Mirror preprocess_image(), but start from an in-memory BGR frame.
        face_aligned, landmarks = align_face(frame.bgr)

        img_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        img_float_norm = cv2.resize(img_rgb, (224, 224))
        img_float_norm = np.float32(img_float_norm) / 255.0

        input_tensor = ImagePreprocessor.preprocess_face(
            cv2.cvtColor(np.uint8(img_float_norm * 255), cv2.COLOR_RGB2BGR),
        ).to(server_state.device)

        _label_str, prob, _reasoning = perform_inference(input_tensor)
        frame_scores.append(float(prob))

        if prob > worst_score:
            worst_score = float(prob)
            worst_idx = frame.index
            worst_input_tensor = input_tensor
            worst_img_float_norm = img_float_norm
            worst_landmarks = landmarks

    video_score = float(np.mean(frame_scores)) if frame_scores else 0.0
    label_str = "FAKE" if video_score > Config.FAKE_PROBABILITY_THRESHOLD else "REAL"

    reasoning = None
    if (
        label_str == "FAKE"
        and worst_input_tensor is not None
        and worst_landmarks is not None
    ):
        grayscale_cam = server_state.cam(input_tensor=worst_input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        reasoning = analyze_heatmap_regions(grayscale_cam, worst_landmarks)

    if worst_input_tensor is None or worst_img_float_norm is None:
        raise HTTPException(status_code=500, detail="Failed to compute video result")

    grad_cam_image = generate_grad_cam(worst_input_tensor, worst_img_float_norm)

    return {
        "label": label_str,
        "score": video_score,
        "reasoning": reasoning,
        "grad_cam_image": grad_cam_image,
        "sampled_frames": len(extracted),
        "worst_frame_index": int(worst_idx),
        "worst_frame_score": float(worst_score),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
