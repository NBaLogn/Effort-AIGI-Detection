# 🎯 Effort Model Fine-Tuning

This guide provides comprehensive instructions for fine-tuning the Effort model using the SVD decomposition approach.

## 🚀 Quick Start

### Commands

### finetune.sh

```bash
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset '[DATASET_PATH]' \
    --test_dataset '[DATASET_PATH]' \
    --pretrained_weights '[PATH_TO]/effort_clip_L14_trainOn_FaceForensic.pth'
```

### eval.sh

```bash
uv run 'DeepfakeBench/training/evaluate_finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights '[PATH_TO_FINETUNED_WEIGHT]'\
    --test_dataset '[DATASET_PATH]' '[DATASET_PATH]' \
    --output_dir '[PATH_TO_OUTPUT_FOLDER]'
```

### infer.sh

```bash
uv run 'DeepfakeBench/training/inference.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --landmark_model \
        '[PATH_TO]/shape_predictor_81_face_landmarks.dat' \
    --weights \
        '[PATH_TO_FINETUNED_WEIGHT]' \
    --image \
        '[PATH_TO_IMAGE_FILE_OR_FOLDER]'
```

## 📋 Fine-Tuning Configuration

The fine-tuning configuration file (`effort_finetune.yaml`) includes optimized settings:

### Settings for batch2k

Used effort_clip_L14_trainOn_FaceForensic.pth, used 2000 extracted faces from Chameleon, Genimage, quan and quanFaceSwap datasets, finetuned for 2 epochs.

### Settings for batchAll

Used effort_clip_L14_trainOn_FaceForensic.pth, used all the extracted faces from Chameleon, Genimage, quan and quanFaceSwap datasets, finetuned for 10 epochs.

#### Current Fine-Tune Configuration

```yaml
# Fine-tuning specific settings
fine_tune: true
pretrained_checkpoint: null
freeze_backbone: true
train_classification_head: true
train_svd_residuals: true

# Training configuration
nEpochs: 10
lr_scheduler: cosine
lr_T_max: 10
lr_eta_min: 0.000001

# Optimizer settings
optimizer:
  type: adam
  adam:
    lr: 0.00005
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001

# Data augmentation settings
data_aug:
  flip_prob: 0.5
  rotate_prob: 0.4
  rotate_limit: [-10, 10]
  blur_prob: 0.3
  brightness_prob: 0.3
  brightness_limit: [-0.1, 0.1]
```

## 🔧 How Fine-Tuning Works

### SVD Decomposition Approach

The Effort model uses **Orthogonal Subspace Decomposition** for efficient fine-tuning:

1. **Original Weight Matrix**: `W = U @ Σ @ Vᵀ`
2. **Fixed Main Components**: `W_main = U_r @ Σ_r @ V_rᵀ` (top r components)
3. **Trainable Residuals**: `W_residual = U_residual @ Σ_residual @ V_residualᵀ` (remaining components)
4. **Total Weight**: `W_total = W_main + W_residual`

### Parameter Efficiency

- **Fixed Parameters**: ~99% of original parameters (preserve pre-trained knowledge)
- **Trainable Parameters**: ~1% of original parameters (SVD residuals + classification head)
- **Total Trainable**: ~1-5% of model parameters

## 📊 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Primary Metrics**: AUC, EER, Accuracy, AP
- **Additional Metrics**: Precision, Recall, F1 Score
- **Detailed Logging**: Per-batch progress, final summary
- **Result Formats**: JSON output for easy analysis

## 🔍 Monitoring and Debugging

### Logging

- **Finetuning Logs**: `training/logs/finetuning.log`
- **Evaluation Logs**: `evaluation_results/evaluation.log`
- **TensorBoard**: Automatic logging of metrics

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/effort-aigi-detection.git
cd effort-aigi-detection
```

### 2. Set Up Python Environment
```bash
# Install Python dependencies using uv
uv sync
```

This will install all Python dependencies listed in `pyproject.toml`, including:
- FastAPI and Uvicorn for the backend server
- PyTorch and related ML libraries
- OpenCV, dlib, and other computer vision tools
- Deepfake detection model dependencies

### 3. Set Up Frontend
```bash
cd frontend
npm install
# or
uv run npm install
```

This will install Next.js and React dependencies.

### 4. Download Required Models
The application requires specific model files:

#### Landmark Detection Model
Download the 81-landmark face shape predictor at https://github.com/codeniko/shape_predictor_81_face_landmarks

#### Deepfake Detection Weights
You need pretrained Effort model weights. Place them in the appropriate location. The file server.py looks for the weight file and landmark file, you must change the path.

## 🚀 Running the Application

### Development Mode

#### 1. Start the Backend Server
```bash
# From the project root
uv run backend/server.py
```

The backend will start on `http://0.0.0.0:8000` with:
- FastAPI REST API for deepfake detection
- CORS enabled for frontend communication
- Automatic model loading and Grad-CAM visualization
- Health check endpoint at `/health`

#### 2. Start the Frontend Development Server
```bash
cd frontend
npm run dev
# or
uv run npm run dev
```

The frontend will start on `http://localhost:3000` with:
- Hot module replacement for instant updates
- Interactive deepfake detection interface
- Image upload and analysis capabilities
- Visual Grad-CAM explanations

## 📊 API Endpoints

### POST /predict
Upload an image for deepfake detection:

**Request:**
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

**Response:**
```json
{
  "label": "FAKE",
  "score": 0.95,
  "reasoning": "Suspicious textures detected around the eyes",
  "grad_cam_image": "data:image/jpeg;base64,..."
}
```

### GET /health
Check if the backend is running:
```bash
curl http://localhost:8000/health
```

## 🎯 Usage

1. **Upload Images**: Drag and drop images or use the file picker
2. **View Results**: See real-time deepfake detection results
3. **Analyze Heatmaps**: Visual Grad-CAM explanations show which facial regions triggered the detection
4. **Batch Processing**: Upload multiple images for batch analysis

##  Project Structure

```
effort-aigi-detection/
├── backend/                  # FastAPI backend
│   ├── server.py             # Main backend application
│   └── gradcam_utils.py      # Grad-CAM utilities
├── frontend/                 # Next.js frontend
│   ├── app/                  # Application pages
│   ├── components/           # React components
│   └── public/               # Static assets
├── DeepfakeBench/            # Core detection logic
│   ├── training/             # Training scripts
│   └── preprocessing/        # Data preprocessing
└── README.md                 # This file
```
