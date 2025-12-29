import sys
from pathlib import Path

# Add DeepfakeBench/training and backend to sys.path to allow imports
current_dir = Path(__file__).resolve().parent
deepfake_bench_path = current_dir.parent / "DeepfakeBench" / "training"
sys.path.insert(0, str(deepfake_bench_path))
sys.path.insert(0, str(current_dir))

import base64
import logging
from contextlib import asynccontextmanager, suppress

import cv2
import numpy as np
import torch
import uvicorn
import yaml

# Import DeepfakeBench modules
# We explicitly check if we are importing the local one
with suppress(ImportError):
    from detectors import DETECTOR

import dlib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradcam_utils import reshape_transform
from inference import DeviceManager, FaceAlignment, ImagePreprocessor
from pydantic import BaseModel

# Import Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    """Wrapper to make EffortDetector compatible with Grad-CAM."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"image": x})["cls"]


# Configuration
CONFIG_PATH = deepfake_bench_path / "config" / "detector" / "effort_finetune.yaml"
WEIGHTS_PATH = Path(
    "DeepfakeBench/training/weights/batchFacesAll.pth",
)
LANDMARK_MODEL_PATH = Path(
    "DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
)

DEVICE = None
FACE_DETECTOR = None

# Landmark group indices for 81-landmark model
LANDMARK_GROUPS = {
    "eyes": list(range(36, 48)),
    "nose": list(range(27, 36)),
    "mouth": list(range(48, 68)),
    "forehead": list(range(68, 81)),
}


def find_best_weights():
    # Use the hardcoded path requested by the user
    if WEIGHTS_PATH.exists():
        logger.info("Found weights at requested path: %s", WEIGHTS_PATH)
        return WEIGHTS_PATH

    logger.warning(
        "Requested weights not found at %s. Falling back to search.",
        WEIGHTS_PATH,
    )

    # Attempt to find the best weights in the logs directory
    # Only looking at 'effort' related folders
    logs_dir = deepfake_bench_path / "logs"
    if not logs_dir.exists():
        return None

    # Simple heuristic: find most recent modified ckpt_best.pth in subdirs
    candidates = list(logs_dir.rglob("ckpt_best.pth"))
    if not candidates:
        return None

    # Sort by modification time, newest first
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    logger.info(f"Found weights: {[str(c) for c in candidates]}")
    return candidates[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CAM, DEVICE

    # Load Device
    DEVICE = DeviceManager.get_optimal_device()
    logger.info("Using device: %s", DEVICE)

    # Load Config
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found at {CONFIG_PATH}")
        raise RuntimeError("Config file not found")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Load Model
    # We need to manually instantiate because we might not have the original training strict structure
    try:
        MODEL = DETECTOR[config["model_name"]](config).to(DEVICE)
    except KeyError as e:
        logger.error("Model %s not found in DETECTOR registry", config["model_name"])
        raise RuntimeError("Model definition not found") from e

    # Load Weights
    weights_path = find_best_weights()
    if weights_path:
        logger.info(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=DEVICE)
        state_dict = ckpt.get("state_dict", ckpt)
        # Remove module. prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        MODEL.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(
            "No checkpoint found! Using random initialization (Expect poor results)",
        )

    MODEL.eval()

    # Setup Grad-CAM
    # Target layer for ViT backbone in Effort model
    # Backbone is clip_model.vision_model
    # We target the last layer norm of the encoder
    target_layers = [MODEL.backbone.encoder.layers[-1].layer_norm1]

    # Wrap model for Grad-CAM
    wrapped_model = ModelWrapper(MODEL)

    CAM = GradCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    # Load Face Detector
    global FACE_DETECTOR
    if LANDMARK_MODEL_PATH.exists():
        logger.info("Loading landmark model from %s", LANDMARK_MODEL_PATH)
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(str(LANDMARK_MODEL_PATH))
        FACE_DETECTOR = FaceAlignment(face_det, shape_predictor)
    else:
        logger.warning(
            "Landmark model not found at %s. Face alignment disabled.",
            LANDMARK_MODEL_PATH,
        )

    yield

    # Clean up
    del MODEL
    del CAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    label: str
    score: float
    reasoning: str | None = None
    grad_cam_image: str  # Base64 encoded image


@app.get("/health")
def health_check():
    return {"status": "ok"}


def analyze_heatmap_regions(mask: np.ndarray, landmarks: np.ndarray) -> str:
    """Analyze which facial regions are most affected by the heatmap."""
    region_scores = {}

    for name, indices in LANDMARK_GROUPS.items():
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

    if intensity < 0.2:
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess with Face Alignment if available
    face_aligned = img_bgr
    landmarks = None
    if FACE_DETECTOR:
        face_result = FACE_DETECTOR.extract_aligned_face(img_bgr, res=224)
        if face_result[0] is not None:
            face_aligned = face_result[0]
            landmarks = face_result[1]
            logger.info("Face aligned for %s", file.filename)
        else:
            logger.warning(
                "No face detected in %s, using original image", file.filename,
            )

    # Just resize to 224x224 for the model input
    # Note: inference.py ImagePreprocessor handles normalization
    img_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)

    # Keep a normalized float copy for Grad-CAM visualization
    img_float_norm = cv2.resize(img_rgb, (224, 224))
    img_float_norm = np.float32(img_float_norm) / 255.0

    # Prepare Tensor
    input_tensor = ImagePreprocessor.preprocess_face(face_aligned).to(DEVICE)

    # Inference
    # We need to compute Grad-CAM and Prediction

    # 1. Prediction
    with torch.no_grad():
        pred_dict = MODEL(
            {"image": input_tensor, "label": torch.tensor([0]).to(DEVICE)},
        )  # Mock label
        prob = pred_dict["prob"].item()
        # prob is probability of FAKE (1).

    label_str = "FAKE" if prob > 0.5 else "REAL"

    # 2. Grad-CAM
    # Target category: we want to see what makes it FAKE (1) or REAL (0)?
    # Usually we visualize the predicted class.
    targets = None  # Uses predicted class by default

    grayscale_cam = CAM(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 3. Reasoning
    reasoning = None
    if label_str == "FAKE" and landmarks is not None:
        reasoning = analyze_heatmap_regions(grayscale_cam, landmarks)

    # Overlay
    cam_image = show_cam_on_image(img_float_norm, grayscale_cam, use_rgb=True)

    # Encode Grad-CAM to Base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    grad_cam_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "label": label_str,
        "score": float(prob),
        "reasoning": reasoning,
        "grad_cam_image": f"data:image/jpeg;base64,{grad_cam_b64}",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
