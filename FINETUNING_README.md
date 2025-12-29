# 🎯 Effort Model Fine-Tuning 

This guide provides comprehensive instructions for fine-tuning the Effort model using the SVD decomposition approach.

## 🚀 Quick Start

### COMMANDS

## finetune.sh

```bash
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset '[DATASET_PATH]' \
    --test_dataset '[DATASET_PATH]' \
    --pretrained_weights '[PATH_TO]/effort_clip_L14_trainOn_FaceForensic.pth'
```

## eval.sh

```bash
uv run 'DeepfakeBench/training/evaluate_finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights '[PATH_TO_FINETUNED_WEIGHT]'\
    --test_dataset '[DATASET_PATH]' '[DATASET_PATH]' \
    --output_dir '[PATH_TO_OUTPUT_FOLDER]'
```  
## infer.sh
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

# settings for batch2k
Used effort_clip_L14_trainOn_FaceForensic.pth, finetuned on 2000 extracted faces from Chameleon, Genimage, quan and quanFaceSwap datasets. The key params was changed below:
### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_batchSize` | 8 | Smaller batch size for stability |
| `optimizer.lr` | 0.0001 | Lower learning rate for fine-tuning |
| `weight_decay` | 0.0001 | Reduced regularization |
| `nEpochs` | 10 | 5-15 epochs typically sufficient |
| `rec_iter` | 50 | Frequent progress logging |
| `freeze_backbone` | true | Freeze SVD main components |
| `train_svd_residuals` | true | Train SVD residual components |

### Data Augmentation (Reduced for Fine-Tuning)

```yaml
data_aug:
  flip_prob: 0.3          # Reduced from 0.5
  rotate_prob: 0.2        # Reduced from 0.5
  rotate_limit: [-5, 5]   # Reduced range
  blur_prob: 0.2          # Reduced from 0.5
  brightness_limit: [-0.05, 0.05]  # Reduced range
```
#### settings for batchAll
Used effort_clip_L14_trainOn_FaceForensic.pth, finetuned on all the extracted faces from Chameleon, Genimage, quan and quanFaceSwap datasets. 
TODO: insert current finetune config

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
- **TensorBoard**: Automatic logging of loss and metrics

## 🔧 Technical Details

### Memory Optimization

The fine-tuning script includes:

- **Automatic device selection** (CUDA/MPS/CPU)
- **Efficient data loading** with proper batching
- **Memory monitoring** for MPS devices
- **Mixed precision support** via `torch.autocast`
