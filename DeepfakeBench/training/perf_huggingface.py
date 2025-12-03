"""
Simplified HuggingFace CLIP inference for deepfake detection.
Usage: uv run python perf_huggingface.py --model openai/clip-vit-base-patch32 --image image.jpg
"""

import argparse
import torch
import cv2
from pathlib import Path
from transformers import CLIPModel, AutoProcessor
from torchvision.transforms import ToPILImage

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available()
                     else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                     else "cpu")
print(f"Using {device}")


def load_model(model_name):
    """Load CLIP model and processor"""
    print(f"Loading {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def preprocess_image(img_path):
    """Load and preprocess image for CLIP"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    # Convert BGR to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Convert to PIL and apply CLIP preprocessing
    pil_img = ToPILImage()(img)
    return pil_img


@torch.no_grad()
def predict_image(model, processor, image):
    """Run inference on a single image"""
    inputs = processor(images=[image], return_tensors="pt").to(device)

    # Get CLIP features
    outputs = model.vision_model(**inputs)
    features = outputs.pooler_output

    # Simple binary classification (random weights - replace with trained model)
    classifier = torch.randn(2, features.shape[-1]).to(device)
    logits = torch.matmul(features, classifier.t())
    probs = torch.softmax(logits, dim=-1)

    pred_class = torch.argmax(logits, dim=-1).item()
    confidence = probs[0, pred_class].item()

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description="HuggingFace CLIP Deepfake Detection")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32",
                       help="HuggingFace model name")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.model)

    # Load and process image
    try:
        image = preprocess_image(args.image)
        pred_class, confidence = predict_image(model, processor, image)

        # Print results
        label = "FAKE" if pred_class == 1 else "REAL"
        print(f"Prediction: {label} (confidence: {confidence:.4f})")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()