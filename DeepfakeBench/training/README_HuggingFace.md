# HuggingFace Model Integration for Deepfake Detection

This document describes the `perf_huggingface.py` script, which is a HuggingFace-based alternative to the original `perf.py` script.

## Key Differences from Original `perf.py`

### 1. Model Loading
- **Original**: Loads local PyTorch model weights from `.pth` files using `torch.load()`
- **HuggingFace**: Loads models directly from HuggingFace Hub using `CLIPModel.from_pretrained()`

### 2. Model Architecture
- **Original**: Uses custom detector classes with local weights and complex SVD modifications
- **HuggingFace**: Uses standard CLIP models from HuggingFace with optional SVD modifications

### 3. Dependencies
- **Original**: Requires local model weights and custom detector implementations
- **HuggingFace**: Requires `transformers` library and internet connection for model downloads

### 4. Classification Head
- **Original**: Uses trained classification head from local weights
- **HuggingFace**: Uses a simple linear layer (requires proper training for production use)

## Usage

### Basic Usage
```bash
uv run python DeepfakeBench/training/perf_huggingface.py \
    --detector_config DeepfakeBench/training/config/detector/huggingface_clip.yaml \
    --weights openai/clip-vit-large-patch14 \
    --image path/to/image.jpg
```

### With Face Detection
```bash
uv run python DeepfakeBench/training/perf_huggingface.py \
    --detector_config DeepfakeBench/training/config/detector/huggingface_clip.yaml \
    --weights openai/clip-vit-large-patch14 \
    --image path/to/image.jpg \
    --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat
```

### Batch Processing
```bash
uv run python DeepfakeBench/training/perf_huggingface.py \
    --detector_config DeepfakeBench/training/config/detector/huggingface_clip.yaml \
    --weights openai/clip-vit-large-patch14 \
    --image path/to/image_directory/
```

## Command Line Arguments

- `--detector_config`: YAML configuration file path (default: "training/config/detector/huggingface_clip.yaml")
- `--weights`: HuggingFace model name (e.g., "openai/clip-vit-large-patch14")
- `--image`: Path to image file or directory containing images (required)
- `--landmark_model`: Path to dlib landmark model for face detection (optional)

## Supported Models

The script supports any CLIP model available on HuggingFace Hub, including:
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-base-patch16`
- `openai/clip-vit-large-patch14`
- `openai/clip-vit-large-patch14-336`

## Output

The script generates `inference_results_huggingface.txt` with:
- Prediction results for each image
- True labels (if directory-based labeling is available)
- AUC and PR-AUC scores (for multi-image evaluation)
- Performance metrics

## Important Notes

1. **Classification Head**: The current implementation uses a random linear layer for binary classification. For production use, you should train or load a proper classification head.

2. **Face Detection**: Face detection and alignment work the same as the original script when `--landmark_model` is provided.

3. **Model Caching**: Models are cached locally after first download, so subsequent runs are faster.

4. **GPU Support**: Automatically detects and uses CUDA/MPS/CPU.

## Integration with Existing Workflow

The HuggingFace version maintains the same interface and output format as the original script, making it a drop-in replacement for evaluation purposes. However, the classification performance may differ due to the untrained classification head.

For production use, consider:
1. Training a proper classification head on the CLIP features
2. Fine-tuning the CLIP model on your specific deepfake detection task
3. Using the SVD modifications from the original implementation for better generalization