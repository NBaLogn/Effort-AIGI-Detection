import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as T
import yaml
from detectors import DETECTOR
from imutils import face_utils
from PIL import Image as pil_image
from skimage import transform as trans
from sklearn.metrics import roc_auc_score

"""
Usage:
    python infer.py \
        --detector_config ./training/config/detector/effort.yaml \
        --weights ../../DeepfakeBenchv2/training/weights/easy_clipl14_cdf.pth \
        --image ./id9_id6_0009.jpg \
        --landmark_model ../../DeepfakeBenchv2/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
"""

# Check for GPU availability with priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")


@torch.no_grad()
def inference(model, data_dict):
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
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)  # FIXME ⚠
    model.eval()
    print("[✓] Detector loaded.")
    return model


def preprocess_face(img_bgr: np.ndarray):
    """BGR → tensor"""
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
    return cls_out, prob


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_image_paths(path_str: str, limit: int = 100) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
        return [p]

    img_list = [
        fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS
    ]
    if not img_list:
        raise RuntimeError(
            f"[Error] No valid image files found in directory: {path_str}"
        )

    # Limit to specified number of images
    img_list = sorted(img_list)[:limit]
    
    return img_list


def extract_true_labels(img_paths: List[Path], base_path: str) -> List[int]:
    """
    Extract true labels (0=Real, 1=Fake) from image paths.
    
    Supports multiple strategies:
    1. Directory-based: If base_path contains 'real' and 'fake' subdirectories
    2. Filename-based: Checks for patterns like 'real', 'fake', 'fake_' in filenames
    
    Returns:
        List of integer labels (0 for Real, 1 for Fake)
    """
    labels = []
    base_path = Path(base_path)
    
    # Strategy 1: Check for directory-based labeling
    # Look for 'real' and 'fake' subdirectories in the base path
    try:
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        real_dirs = [d for d in subdirs if 'real' in d.name.lower()]
        fake_dirs = [d for d in subdirs if 'fake' in d.name.lower() or 'synthetic' in d.name.lower()]
        
        if real_dirs or fake_dirs:
            # Directory-based labeling
            for img_path in img_paths:
                # Check if the image is in a real or fake subdirectory
                img_parent = img_path.parent
                if any(real_dir in img_path.parents for real_dir in real_dirs):
                    labels.append(0)  # Real
                elif any(fake_dir in img_path.parents for fake_dir in fake_dirs):
                    labels.append(1)  # Fake
                else:
                    # If we can't determine from directory, try filename
                    labels.append(_extract_label_from_filename(img_path.name))
            return labels
    except:
        pass  # If directory structure check fails, try filename-based approach
    
    # Strategy 2: Filename-based labeling for all images
    for img_path in img_paths:
        labels.append(_extract_label_from_filename(img_path.name))
    
    return labels


def _extract_label_from_filename(filename: str) -> int:
    """
    Extract label from filename using common patterns.
    Returns 0 for Real, 1 for Fake.
    """
    filename_lower = filename.lower()
    
    # Common patterns indicating fake images
    fake_patterns = ['fake', 'synthetic', 'generated', 'ai_', '_fake', 'synth']
    real_patterns = ['real', 'original', 'authentic', 'genuine']
    
    # Check for fake patterns first (more specific)
    for pattern in fake_patterns:
        if pattern in filename_lower:
            return 1  # Fake
    
    # Check for real patterns
    for pattern in real_patterns:
        if pattern in filename_lower:
            return 0  # Real
    
    # If no pattern matches, assume real (conservative approach)
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake image inference (single image version)"
    )
    p.add_argument(
        "--detector_config",
        default="training/config/detector/effort.yaml",
        help="YAML 配置文件路径",
    )
    p.add_argument("--weights", required=True, help="Detector 预训练权重")
    p.add_argument("--image", required=True, help="tested image")
    p.add_argument(
        "--landmark_model",
        default=False,
        help="dlib 81 landmarks dat 文件 / 如果不需要裁剪人脸就是False",
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

    img_paths = collect_image_paths(args.image)
    multiple = len(img_paths) > 1
    if multiple:
        print(f"Collected {len(img_paths)} images in total，let's infer them...\n")

    # ---------- infer ----------
    predictions = []
    probabilities = []
    valid_paths = []
    
    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warning] loading wrong，skip: {img_path}", file=sys.stderr)
            continue

        cls, prob = infer_single_image(img, face_det, shape_predictor, model)
        
        # Store predictions and probabilities for AUC calculation
        predictions.append(int(cls))
        probabilities.append(float(prob))
        valid_paths.append(img_path)
        
        print(
            f"[{idx}/{len(img_paths)}] {img_path.name:>30} | Pred Label: {cls} "
            f"(0=Real, 1=Fake) | Fake Prob: {prob:.4f}"
        )
    
    # ---------- Calculate AUC if we have multiple images ----------
    if len(predictions) > 1:
        try:
            # Extract true labels
            true_labels = extract_true_labels(valid_paths, args.image)
            
            # Calculate AUC
            auc_score = roc_auc_score(true_labels, probabilities)
            
            print(f"\n" + "="*50)
            print(f"PERFORMANCE METRICS")
            print(f"="*50)
            print(f"Total images processed: {len(predictions)}")
            print(f"AUC (Area Under Curve): {auc_score:.4f}")
            print(f"="*50)
            
        except Exception as e:
            print(f"[Warning] Could not calculate AUC: {e}", file=sys.stderr)
            print("AUC calculation may require:")
            print("- Images with 'real' or 'fake' in directory names")
            print("- Filenames containing 'real', 'fake', 'synthetic', etc.")
    elif len(predictions) == 1:
        print(f"\nSingle image processed - AUC calculation not applicable.")


if __name__ == "__main__":
    main()
