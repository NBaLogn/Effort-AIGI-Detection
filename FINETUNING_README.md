# 🎯 Effort Model Fine-Tuning Guide

This guide provides comprehensive instructions for fine-tuning the Effort model using the SVD decomposition approach.

## 🚀 Quick Start

### 1. Basic Fine-Tuning Command

```bash
uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset UADFV \
    --test_dataset UADFV \
    --pretrained_weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort/effort_clip_L14_trainOn_UniversalFakeDetect.pth
```

### 2. Evaluation Command

```bash
uv run DeepfakeBench/training/evaluate_finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --weights /path/to/finetuned_weights.pth \
    --test_dataset UADFV Celeb-DF-v2 \
    --output_dir evaluation_results
```

## 📋 Fine-Tuning Configuration

The fine-tuning configuration file (`effort_finetune.yaml`) includes optimized settings:

### Key Parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_batchSize` | 8 | Smaller batch size for stability |
| `optimizer.lr` | 0.0001 | Lower learning rate for fine-tuning |
| `weight_decay` | 0.0001 | Reduced regularization |
| `nEpochs` | 10 | 5-15 epochs typically sufficient |
| `rec_iter` | 50 | Frequent progress logging |
| `freeze_backbone` | true | Freeze SVD main components |
| `train_svd_residuals` | true | Train SVD residual components |

### Data Augmentation (Reduced for Fine-Tuning):

```yaml
data_aug:
  flip_prob: 0.3          # Reduced from 0.5
  rotate_prob: 0.2        # Reduced from 0.5
  rotate_limit: [-5, 5]   # Reduced range
  blur_prob: 0.2          # Reduced from 0.5
  brightness_limit: [-0.05, 0.05]  # Reduced range
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

## 🎛️ Advanced Fine-Tuning Options

### 1. Partial Unfreezing

To unfreeze specific layers for more aggressive fine-tuning:

```python
# In finetune.py, modify configure_fine_tuning():
for name, param in model.named_parameters():
    if 'backbone.layer' in name and 'attention' in name:
        param.requires_grad = True  # Unfreeze attention layers
```

### 2. Learning Rate Scheduling

Add to your config:

```yaml
lr_scheduler: "cosine"
lr_T_max: 10
lr_eta_min: 1e-6
```

### 3. Gradient Accumulation

For larger effective batch sizes:

```python
# Modify the training loop in finetune.py
accumulation_steps = 4
for batch_idx, data_dict in enumerate(train_data_loader):
    # Forward pass
    losses, predictions = trainer.train_step(data_dict)

    # Accumulate gradients
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📊 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Primary Metrics**: AUC, EER, Accuracy, AP
- **Additional Metrics**: Precision, Recall, F1 Score
- **Detailed Logging**: Per-batch progress, final summary
- **Result Formats**: JSON output for easy analysis

## 🔍 Monitoring and Debugging

### Logging

- **Training Logs**: `training/logs/finetuning.log`
- **Evaluation Logs**: `evaluation_results/evaluation.log`
- **TensorBoard**: Automatic logging of loss and metrics

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `train_batchSize` to 4 or 2
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Convergence**:
   - Try higher learning rate (1e-3)
   - Unfreeze more layers
   - Increase epochs to 15-20

3. **Overfitting**:
   - Reduce data augmentation
   - Add more regularization
   - Use early stopping

## 🎯 Best Practices

### 1. Dataset Selection

- Start with **small, targeted datasets** for initial fine-tuning
- Gradually increase dataset size and diversity
- Use **validation sets** to monitor generalization

### 2. Learning Rate Strategy

- **Stage 1**: Fine-tune classification head only (1-2 epochs, higher LR)
- **Stage 2**: Fine-tune SVD residuals (3-5 epochs, lower LR)
- **Stage 3**: Optional unfreezing (2-3 epochs, very low LR)

### 3. Evaluation Strategy

- Evaluate on **multiple diverse datasets**
- Monitor **both in-domain and out-of-domain** performance
- Track **metric trends** across epochs

## 📚 Example Workflows

### Workflow 1: Basic Fine-Tuning

```bash
# Step 1: Fine-tune on target dataset
uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset UADFV \
    --test_dataset UADFV \
    --pretrained_weights pretrained/effort_base.pth

# Step 2: Evaluate on multiple datasets
uv run DeepfakeBench/training/evaluate_finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --weights finetuned/effort_uadfv.pth \
    --test_dataset UADFV Celeb-DF-v2 FaceForensics++ \
    --output_dir evaluation_results
```

### Workflow 2: Multi-Stage Fine-Tuning

```bash
# Stage 1: Classification head only
uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset UADFV \
    --nEpochs 2 \
    --optimizer.lr 0.001

# Stage 2: Full fine-tuning
uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset UADFV \
    --nEpochs 8 \
    --pretrained_weights stage1_checkpoint.pth
```

## 🔧 Technical Details

### SVD Implementation

The SVD decomposition is implemented in [`effort_detector.py`](DeepfakeBench/training/detectors/effort_detector.py):

- **Line 337**: `torch.linalg.svd()` call
- **Lines 343-364**: Component separation
- **Lines 306-312**: Gradient configuration
- **Lines 212-216**: Forward pass reconstruction

### Memory Optimization

The fine-tuning script includes:

- **Automatic device selection** (CUDA/MPS/CPU)
- **Efficient data loading** with proper batching
- **Memory monitoring** for MPS devices
- **Mixed precision support** via `torch.autocast`

## 📈 Expected Results

With proper fine-tuning, you should see:

- **Fast convergence**: 80%+ of final performance in first 3-5 epochs
- **Good generalization**: Maintained performance on unseen datasets
- **Efficiency**: 5-10x faster than full model fine-tuning
- **Stability**: Consistent results across multiple runs

## 🎓 Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| **Low accuracy** | Increase epochs, adjust learning rate, check data quality |
| **Overfitting** | Reduce data augmentation, add regularization, use early stopping |
| **Slow training** | Reduce batch size, use mixed precision, check device utilization |
| **Memory errors** | Reduce batch size, use gradient accumulation, enable CPU offloading |
| **NaN losses** | Reduce learning rate, check data normalization, verify label correctness |

## 📚 References

- **Original Paper**: [Effort: Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2411.15633)
- **SVD Implementation**: [`effort_detector.py:316-378`](DeepfakeBench/training/detectors/effort_detector.py:316)
- **Training Infrastructure**: [`trainer/trainer.py`](DeepfakeBench/training/trainer/trainer.py)

This fine-tuning implementation leverages the unique SVD decomposition approach of the Effort model to provide efficient, stable, and generalizable fine-tuning capabilities.