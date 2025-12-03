# Simplified HuggingFace CLIP Inference for Deepfake Detection

This document describes the simplified `perf_huggingface.py` script, which provides basic HuggingFace CLIP model inference for deepfake detection.

## Key Features

- **Simple Interface**: Direct model loading from HuggingFace Hub
- **CLIP Models**: Support for any CLIP vision model (ViT-Base, ViT-Large, etc.)
- **GPU Support**: Automatic MPS/CUDA/CPU detection
- **Easy Usage**: Minimal dependencies and straightforward API

## Differences from Original `perf.py`

- **Simplified**: Removed complex face detection, directory analysis, and batch processing
- **Direct**: Loads models directly without detector registry system
- **Basic**: Uses random classification weights (for demonstration only)
- **Focused**: Single image inference only

## Usage

### Basic Usage
```bash
uv run python DeepfakeBench/training/perf_huggingface.py \
    --model openai/clip-vit-base-patch32 \
    --image path/to/image.jpg
```

### Different Models
```bash
# Use larger model
uv run python DeepfakeBench/training/perf_huggingface.py \
    --model openai/clip-vit-large-patch14 \
    --image path/to/image.jpg
```

## Command Line Arguments

- `--model`: HuggingFace model name (default: "openai/clip-vit-base-patch32")
- `--image`: Path to image file (required)

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