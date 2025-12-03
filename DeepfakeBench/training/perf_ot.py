import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from detectors import DETECTOR
from imutils import face_utils
from PIL import Image as pil_image
from skimage import transform as trans
from sklearn.metrics import roc_auc_score, average_precision_score

# Constants
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_RES = 224
MAX_DIR_LEVELS = 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

"""
Usage:
    python perf_ot.py \
        --detector_config ./training/config/detector/effort.yaml \
        --weights ../../DeepfakeBenchv2/training/weights/easy_clipl14_cdf.pth \
        --image ./id9_id6_0009.jpg \
        --landmark_model ../../DeepfakeBenchv2/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat

    # For HuggingFace models:
    python perf_ot.py \
        --detector_config ./training/config/detector/huggingface_clip.yaml \
        --weights openai/clip-vit-large-patch14 \
        --image ./id9_id6_0009.jpg \
        --landmark_model ../../DeepfakeBenchv2/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
"""

# Check for GPU availability with priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA (NVIDIA GPU)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")


@torch.no_grad()
def inference(model, data_dict):
    """Inference function compatible with detector interface"""
    data, label = data_dict["image"], data_dict["label"]
    # move data to GPU
    data_dict["image"], data_dict["label"] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


# preprocess the input image --> cropped face, resize = 256, adding a dimension of batch (output shape: 1x3x256x256)
def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)

    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
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

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)

        # Check if faces were detected in the cropped image
        if len(face_align) > 0:
            landmark = predictor(cropped_face, face_align[0])
            landmark = face_utils.shape_to_np(landmark)
        else:
            # No face detected in cropped image, return without landmarks
            landmark = None

        return cropped_face, landmark, face

    else:
        return None, None


def load_detector(detector_cfg: str, weights: str):
    """Load detector using flexible configuration loading"""
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert config dict to object with attributes for compatibility
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = Config(cfg)

    # Handle different model types
    model_name = cfg.get("model_name", "effort")

    if model_name == "huggingface_clip":
        # For HuggingFace models, override the model name with weights argument
        if weights:
            config.huggingface_model_name = weights
    else:
        # For traditional models like effort, load weights normally
        if weights:
            ckpt = torch.load(weights, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}

    model_cls = DETECTOR[model_name]
    model = model_cls(config).to(device)

    if model_name != "huggingface_clip":
        # Load traditional weights for non-HuggingFace models
        model.load_state_dict(state, strict=False)  # FIXME ⚠

    model.eval()
    logger.info(f"Detector '{model_name}' loaded successfully.")
    return model


def preprocess_face(img_bgr: np.ndarray):
    """BGR → tensor (same as original)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×H×W


@torch.inference_mode()
def infer_single_image(
    img_bgr: np.ndarray,
    face_detector,
    landmark_predictor,
    model,
) -> Tuple[int, float]:
    """Return (cls_out, prob)"""
    if face_detector is None or landmark_predictor is None:
        face_aligned = img_bgr
    else:
        # Try to extract and align face
        face_detection_result = extract_aligned_face_dlib(
            face_detector, landmark_predictor, img_bgr, res=224
        )

        if len(face_detection_result) == 3 and face_detection_result[0] is not None:
            # Face detected successfully - use the aligned face
            face_aligned, _, _ = face_detection_result
        else:
            # No face detected - fall back to using original image
            face_aligned = img_bgr

    face_tensor = preprocess_face(face_aligned).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    cls_out = preds["cls"].squeeze().cpu().numpy()  # 0/1
    prob = preds["prob"].squeeze().cpu().numpy()  # prob

    # Handle numpy arrays - convert to scalars
    if isinstance(cls_out, np.ndarray):
        cls_out = cls_out.item() if cls_out.ndim == 0 else int(np.argmax(cls_out))
    if isinstance(prob, np.ndarray):
        prob = prob.item() if prob.ndim == 0 else float(prob)

    return cls_out, prob


def collect_image_paths(path_str: str, limit: int = 100) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
        return [p]

    # First, try to find images in the root directory
    img_list = [
        fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS
    ]

    # If no images found in root, search recursively in subdirectories
    if not img_list:
        logger.debug("No images found in root directory, searching subdirectories...")
        img_list = [
            fp for fp in p.rglob("*")
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS
        ]
        logger.debug(f"Found {len(img_list)} images in subdirectories")

    if not img_list:
        raise RuntimeError(
            f"[Error] No valid image files found in directory: {path_str}"
        )

    # Randomly sample to limit number of images for better representation
    if len(img_list) > limit:
        logger.debug(f"Randomly sampling {limit} images from {len(img_list)} available images...")
        img_list = random.sample(img_list, limit)
    else:
        # Shuffle to ensure random order even when taking all images
        random.shuffle(img_list)

    return img_list


def _check_directory_label(dir_name: str, real_dirs: set, fake_dirs: set) -> int | None:
    """Check if a directory name matches real or fake patterns. Returns label or None if no match."""
    if dir_name in real_dirs:
        return 0  # Real
    elif dir_name in fake_dirs:
        return 1  # Fake
    return None


def find_real_fake_dirs(base_path: Path) -> Tuple[set, set]:
    """Find directories containing 'real' or 'fake' patterns."""
    if not base_path.exists():
        logger.warning(f"Base path does not exist: {base_path}")
        return set(), set()

    try:
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        logger.debug(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")

        real_dirs = [d for d in subdirs if 'real' in d.name.lower()]
        fake_dirs = [d for d in subdirs if any(keyword in d.name.lower()
                                                for keyword in ['fake', 'synthetic', 'faceswap'])]

        logger.debug(f"Real directories: {[d.name for d in real_dirs]}")
        logger.debug(f"Fake directories: {[d.name for d in fake_dirs]}")

        return {d.name.lower() for d in real_dirs}, {d.name.lower() for d in fake_dirs}
    except Exception as e:
        logger.error(f"Directory-based labeling failed: {e}")
        return set(), set()


def get_directory_names_at_levels(img_path: Path) -> List[str]:
    """Get directory names at different levels from the image path."""
    dir_names = []
    for i in range(1, MAX_DIR_LEVELS + 1):
        if len(img_path.parts) >= i:
            dir_names.append(img_path.parts[-i].lower())
        else:
            dir_names.append("")
    return dir_names


def determine_label_for_image(img_path: Path, real_dir_names: set, fake_dir_names: set) -> int | None:
    """Determine the label for a single image based on directory names."""
    dir_names = get_directory_names_at_levels(img_path)

    for level, dir_name in enumerate(dir_names, 1):
        if not dir_name:
            continue

        label = _check_directory_label(dir_name, real_dir_names, fake_dir_names)
        if label is not None:
            level_desc = f"from level {level}: {img_path.parts[-level] if len(img_path.parts) >= level else 'N/A'}"
            logger.debug(f"{img_path.name} -> {'REAL' if label == 0 else 'FAKE'} ({level_desc})")
            return label

    logger.debug(f"{img_path.name} -> SKIPPED (no matching directory found)")
    return None


def extract_true_labels(img_paths: List[Path], base_path: str) -> List[int]:
    """
    Extract true labels (0=Real, 1=Fake) from image paths using directory-based labeling only.

    This function only uses directory-based labeling and does not fall back to filename patterns.
    Images are labeled based on their parent directory names containing 'real' or 'fake' patterns.

    Returns:
        List of integer labels (0 for Real, 1 for Fake)
    """
    labels = []
    base_path = Path(base_path)

    logger.debug(f"Analyzing {len(img_paths)} images for label extraction...")
    logger.debug(f"Base path: {base_path}")

    real_dir_names, fake_dir_names = find_real_fake_dirs(base_path)
    if not (real_dir_names or fake_dir_names):
        logger.debug("No 'real' or 'fake' directories found")
        return []

    # Process each image
    real_count = 0
    fake_count = 0
    skipped_count = 0
    path_pattern_counts = {}

    for img_path in img_paths:
        dir_names = get_directory_names_at_levels(img_path)
        path_pattern = "/".join(reversed(dir_names))
        path_pattern_counts[path_pattern] = path_pattern_counts.get(path_pattern, 0) + 1

        label = determine_label_for_image(img_path, real_dir_names, fake_dir_names)
        if label is not None:
            labels.append(label)
            if label == 0:
                real_count += 1
            else:
                fake_count += 1
        else:
            skipped_count += 1

    logger.debug(f"Directory analysis results: Real={real_count}, Fake={fake_count}, Skipped={skipped_count}")
    logger.debug(f"Sample filenames: {[img.name for img in img_paths[:5]]}")
    logger.debug("Path pattern distribution (top 10):")
    sorted_patterns = sorted(path_pattern_counts.items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:10]:
        logger.debug(f"  {pattern}: {count} images")

    return labels


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake image inference (universal model version)"
    )
    p.add_argument(
        "--detector_config",
        required=True,
        help="YAML configuration file path for the detector",
    )
    p.add_argument("--weights", help="Path to model weights (for traditional models) or HuggingFace model name")
    p.add_argument("--image", required=True, help="Path to tested image or directory")
    p.add_argument(
        "--landmark_model",
        default=False,
        help="Path to dlib 81 landmarks dat file / False if no face cropping needed",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of images to process (default: 100)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model = load_detector(args.detector_config, args.weights)
    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(args.landmark_model)
    else:
        face_det, shape_predictor = None, None

    img_paths = collect_image_paths(args.image, args.limit)
    multiple = len(img_paths) > 1
    if multiple:
        logger.info(f"Collected {len(img_paths)} images in total, starting inference...\n")

        # Extract true labels first for the results display
        true_labels = extract_true_labels(img_paths, args.image)

    # ---------- infer ----------
    predictions = []
    probabilities = []
    valid_paths = []
    valid_true_labels = []

    # Open output file for writing results
    output_file = "inference_results_ot.txt"
    with open(output_file, "w") as f:
        f.write("Deepfake Detection Results (Other Models)\n")
        f.write("="*80 + "\n\n")

        for idx, img_path in enumerate(img_paths, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image, skipping: {img_path}")
                continue

            cls, prob = infer_single_image(img, face_det, shape_predictor, model)

            # Store predictions and probabilities for AUC calculation
            predictions.append(int(cls))
            probabilities.append(float(prob))
            valid_paths.append(img_path)

            # Get true label for this image (align with valid images)
            true_label = "N/A"
            if multiple:
                # Find the corresponding true label by matching the path
                for i, path in enumerate(img_paths):
                    if path == img_path and i < len(true_labels):
                        true_label = "REAL" if true_labels[i] == 0 else "FAKE"
                        valid_true_labels.append(true_labels[i])
                        break

            # Write to file with true label and full path
            f.write(f"[{idx}/{len(img_paths)}] {img_path.name:>30} | True: {true_label:>4} | Pred: {cls} "
                    f"(0=Real, 1=Fake) | Prob: {prob:.4f} | Path: {img_path}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*80 + "\n")

        # ---------- Calculate AUC if we have multiple images ----------
        if len(predictions) > 1:
            try:
                # Use the valid true labels that were collected during processing
                true_labels_for_auc = valid_true_labels

                # Write true labels for verification
                f.write("\nTRUE LABELS (for verification):\n")
                for i, (img_path, true_label) in enumerate(zip(valid_paths, true_labels_for_auc), 1):
                    label_str = "REAL" if true_label == 0 else "FAKE"
                    f.write(f"[{i}] {img_path.name:>30} -> {label_str}\n")

                # Check if we have both classes for AUC calculation
                unique_labels = set(true_labels_for_auc)
                f.write(f"\nUnique true labels found: {unique_labels}\n")

                if len(unique_labels) < 2:
                    f.write("\n[WARNING] Cannot calculate AUC - only one class present in true labels\n")
                    f.write(f"Classes found: {list(unique_labels)}\n")
                    f.write(f"This means the dataset appears to contain only {'REAL' if 0 in unique_labels else 'FAKE'} images\n")
                    f.write("\nFor AUC calculation, the dataset should contain both:\n")
                    f.write("- Images with 'real' or 'fake' in directory names\n")

                    # Show some sample filenames for diagnosis
                    f.write("\nSample filenames from your dataset:\n")
                    for i, img_path in enumerate(valid_paths[:5]):
                        f.write(f"  {img_path.name}\n")

                else:
                    # Calculate AUC and PR-AUC
                    auc_score = roc_auc_score(true_labels_for_auc, probabilities)
                    pr_auc_score = average_precision_score(true_labels_for_auc, probabilities)

                    f.write(f"\nTotal images processed: {len(predictions)}\n")
                    f.write(f"Real images: {sum(1 for label in true_labels_for_auc if label == 0)}\n")
                    f.write(f"Fake images: {sum(1 for label in true_labels_for_auc if label == 1)}\n")
                    f.write(f"AUC (Area Under ROC Curve): {auc_score:.4f}\n")
                    f.write(f"PR-AUC (Area Under Precision-Recall Curve): {pr_auc_score:.4f}\n")

                    logger.info(f"AUC (Area Under ROC Curve): {auc_score:.4f}")
                    logger.info(f"PR-AUC (Area Under Precision-Recall Curve): {pr_auc_score:.4f}")

            except Exception as e:
                f.write(f"\n[Warning] Could not calculate AUC: {e}\n")
                f.write("AUC calculation may require:\n")
                f.write("- Images with 'real' or 'fake' in directory names\n")
        elif len(predictions) == 1:
            f.write("\nSingle image processed - AUC calculation not applicable.\n")

    logger.info(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()