# Project Context

## Purpose
**Effort: Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection**

This project implements "Effort" — an SVD-based residual modeling approach for detecting AI-generated images (AIGIs) and deepfakes. The method can be plug-and-play inserted into any ViT-based large models such as CLIP. The project includes:

- **Research/ML Component**: Training and evaluation pipeline built on DeepfakeBench for detecting deepfakes and AI-generated images
- **Web Application**: Frontend/backend system for real-time inference with Grad-CAM visualization
- **Core Innovation**: Decomposes model weights via SVD, freezing low-rank main components while training residual components for better generalization

The project supports both face deepfake detection and general AI-generated image detection across multiple datasets (FaceForensics++, Celeb-DF, GenImage, Chameleon, DF40, etc.).

## Tech Stack

### Backend & ML
- **Python**: 3.13+ (managed via `uv` and `pyproject.toml`)
- **Deep Learning**: PyTorch 2.9.1+, torchvision, torchaudio
- **ML Frameworks**: 
  - CLIP (from OpenAI GitHub)
  - transformers, timm
  - pytorch-grad-cam (for visualization)
- **Computer Vision**: OpenCV, PIL/Pillow, dlib, mediapipe, insightface
- **API Framework**: FastAPI with uvicorn
- **Data Processing**: numpy, pandas, scikit-learn, scikit-image
- **Configuration**: PyYAML

### Frontend
- **Framework**: Next.js 16.1.0
- **UI Library**: React 19.2.3
- **Language**: TypeScript 5+
- **Build Tool**: Next.js built-in (Webpack/Turbopack)

### Development Tools
- **Package Management**: 
  - Python: `uv` (pyproject.toml)
  - Node.js: npm (package.json)
- **Logging**: Python logging module, tensorboard
- **Data Augmentation**: albumentations, imgaug, kornia

## Project Conventions

### Code Style

**Python:**
- Standard library imports first, then third-party, then local imports
- Use type hints where appropriate (especially in newer code)
- Follow PEP 8 conventions
- Use descriptive variable names
- Comments for complex logic (e.g., SVD decomposition, face alignment)
- Path handling: Prefer `pathlib.Path` over string paths where possible
- Logging: Use Python's `logging` module with appropriate levels (INFO, WARNING, ERROR)

**TypeScript/React:**
- Use functional components with hooks
- TypeScript interfaces for props and data structures
- Client components marked with `"use client"` directive
- Modern React patterns (useState, async/await)

**File Naming:**
- Python: snake_case (e.g., `effort_detector.py`, `gradcam_utils.py`)
- TypeScript/React: PascalCase for components (e.g., `Dropzone.tsx`)
- Shell scripts: lowercase with underscores (e.g., `eval.sh`, `infer.sh`)

**Configuration:**
- YAML files for model and training configurations
- Config files located in `DeepfakeBench/training/config/`
- Use descriptive keys with comments where helpful

### Architecture Patterns

**Three-Tier Architecture:**
1. **DeepfakeBench Layer** (`DeepfakeBench/`): Core ML training/evaluation infrastructure
   - `preprocessing/`: Data preprocessing and face detection
   - `training/`: Model training, evaluation, and inference scripts
   - `training/detectors/`: Model implementations (Effort detector)
   - `training/config/`: YAML configuration files

2. **Backend API** (`backend/`): FastAPI server for inference
   - RESTful API endpoints (`/predict`, `/health`)
   - Model loading and lifecycle management
   - Face detection and alignment integration
   - Grad-CAM visualization generation

3. **Frontend** (`frontend/`): Next.js web application
   - Client-side file upload and processing
   - Results display with interactive Grad-CAM overlays
   - Modern React component architecture

**Key Patterns:**
- **Registry Pattern**: Detector models registered via `DETECTOR` registry
- **Configuration-Driven**: Training and inference controlled via YAML configs
- **Device Abstraction**: `DeviceManager` handles CPU/GPU/MPS device selection
- **Modular Preprocessing**: Separate classes for image preprocessing and face alignment
- **Lifespan Management**: FastAPI lifespan context for model loading/unloading

**Data Flow:**
- Training: Dataset JSON → DataLoader → Model → Trainer → Checkpoints
- Inference: Image → Face Detection → Preprocessing → Model → Grad-CAM → Response

### Testing Strategy

**Current State:**
- Manual testing via shell scripts (`eval.sh`, `infer.sh`, `finetune.sh`)
- Evaluation scripts output JSON results to `evaluation_results/`
- Training logs saved to `DeepfakeBench/training/logs/`

**Testing Approach:**
- Use evaluation scripts to validate model performance on test datasets
- Checkpoint comparison utilities in `utils/compare_checkpoints*.py`
- Log analysis tools for training monitoring
- Health check endpoint (`/health`) for API validation

**Recommended Practices:**
- Validate model checkpoints before deployment
- Test inference pipeline with diverse image inputs
- Monitor evaluation metrics (AUC, accuracy, EER) across datasets

### Git Workflow

**Branching:**
- Main branch for stable releases
- Feature branches for new capabilities
- Development branches for experimental work

**Commit Conventions:**
- Descriptive commit messages
- Reference related issues/PRs when applicable
- Separate commits for different logical changes

**File Organization:**
- Keep large model checkpoints and datasets out of git (use `.gitignore`)
- Commit configuration files and code changes
- Evaluation results in `evaluation_results/` are tracked for reference

## Domain Context

**Deepfake/AIGI Detection:**
- **Goal**: Distinguish between real and AI-generated/manipulated images
- **Challenges**: Generalization across different generation methods, compression levels, and datasets
- **Effort Method**: Uses SVD to decompose weights into main (frozen) and residual (trainable) components, improving generalization

**Key Concepts:**
- **Face Deepfakes**: Manipulated face images/videos (FaceForensics++, Celeb-DF, DF40)
- **AIGI Detection**: General AI-generated images (GenImage, Chameleon datasets)
- **Face Alignment**: Using dlib shape predictors to extract and align facial regions
- **Grad-CAM**: Gradient-weighted Class Activation Mapping for model interpretability

**Dataset Structure:**
- Datasets organized with JSON manifests in `DeepfakeBench/preprocessing/dataset_json/`
- Face data stored in `DeepfakeBench/facedata/` with subdirectories per dataset
- Train/val/test splits maintained per dataset

**Model Architecture:**
- Base: CLIP Vision Transformer (ViT-L/14)
- Effort Detector: Wraps CLIP with SVD-based residual modeling
- Output: Binary classification (REAL/FAKE) with probability scores

**Evaluation Metrics:**
- AUC (Area Under Curve) - primary metric
- Accuracy, EER (Equal Error Rate), AP (Average Precision)
- Results logged per dataset and combined test sets

## Important Constraints

**Technical Constraints:**
- **GPU Required**: Training requires CUDA-capable GPU or Apple Silicon (MPS)
- **Memory**: Large models (CLIP-L/14) require significant VRAM/RAM
- **Face Detection Models**: Requires dlib shape predictor file (`shape_predictor_81_face_landmarks.dat`) for face alignment
- **Python Version**: Requires Python 3.13+ (as specified in pyproject.toml)
- **Hardcoded Paths**: Some paths are currently hardcoded (e.g., in `train.py`, `server.py`) - should be made configurable

**Data Constraints:**
- Large datasets (FaceForensics++, DF40) require significant storage
- Preprocessed face crops recommended for faster training
- LMDB format supported for efficient data loading

**Model Constraints:**
- Effort method designed for ViT-based architectures
- Checkpoint compatibility: Must handle both `state_dict` and full checkpoint formats
- Model loading: Handles `module.` prefix removal for DDP-trained models

**Deployment Constraints:**
- Backend requires model weights to be available at specified paths
- Face detection models must be downloaded separately
- CORS configured for local development (currently allows all origins)

## External Dependencies

**Key External Services/Systems:**
- **CLIP Repository**: Git dependency from `https://github.com/openai/CLIP.git`
- **DeepfakeBench**: Codebase structure and benchmarking protocols based on DeepfakeBench
- **DF40 Dataset**: External dataset with 40 distinct forgery methods
- **Google Drive**: Checkpoint storage and distribution (pretrained models)

**Model Checkpoints:**
- FaceForensics++ trained checkpoint (for face deepfake detection)
- GenImage (SDv1.4) trained checkpoint (for general AIGI detection)
- Chameleon (SDv1.4) trained checkpoint (alternative AIGI detection)

**Face Detection Models:**
- dlib shape predictor: `shape_predictor_81_face_landmarks.dat` (81-point facial landmarks)
- Alternative: MediaPipe, RetinaFace, YOLO (utilities available in `utils/`)

**Dataset Sources:**
- FaceForensics++: Face manipulation dataset
- Celeb-DF: High-quality deepfake dataset
- GenImage: AI-generated image dataset
- Chameleon: Alternative AIGI dataset
- DF40: Comprehensive 40-method deepfake benchmark

**Development Dependencies:**
- TensorBoard: Training visualization
- Evaluation tools: Custom scripts in `utils/` for analysis and comparison
