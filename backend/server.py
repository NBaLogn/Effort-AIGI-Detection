"""Deepfake Detection Server.
 
This module provides a FastAPI server for deepfake detection using the Effort model.
It includes endpoints for health checks and image prediction with Grad-CAM visualization.
"""

import base64
import logging
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
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradcam_utils import reshape_transform
from pydantic import BaseModel

# Import Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
        "DeepfakeBench/training/finetuned_weights/batchFacesAll.pth",
    )
    LANDMARK_MODEL_PATH = Path(
        "DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
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


class ServerState:
    """Global server state management."""

    def __init__(self) -> None:
        """Initialize server state."""
        self.device = None
        self.model = None
        self.cam = None
        self.face_detector = None

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "cam") and self.cam is not None:
            del self.cam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        logger.exception("Model %s not found in DETECTOR registry", config["model_name"])
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
        top_region, f"The model is focused on the {top_region} region.",
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

    # Preprocess with Face Alignment if available
    face_aligned = img_bgr
    landmarks = None
    if server_state.face_detector:
        face_result = server_state.face_detector.extract_aligned_face(img_bgr, res=224)
        if face_result[0] is not None:
            face_aligned = face_result[0]
            landmarks = face_result[1]
            logger.info("Face aligned successfully")
        else:
            logger.warning("No face detected, using original image")

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
