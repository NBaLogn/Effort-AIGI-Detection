#!/bin/bash
# Effort Model Training and Fine-Tuning Scripts

# =============================================
# 1. ORIGINAL TRAINING (Example - Commented Out)
# =============================================
# uv run /Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/mac_train.py \
#     --detector_path /Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml \
#     --train_dataset UADFV \
#     --test_dataset UADFV

# =============================================
# 2. FINE-TUNING ON UADFV DATASET
# =============================================
uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset UADFV \
    --test_dataset UADFV \
    --pretrained_weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort_clip_L14_trainOn_FaceForensic.pth

# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================
uv run DeepfakeBench/training/evaluate_finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --weights ./training/weights/finetuned_effort_uadfv.pth \
    --test_dataset UADFV Celeb-DF-v2 \
    --output_dir evaluation_results

# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================
# uv run DeepfakeBench/training/perf_final.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat \
#     --weights ./training/weights/finetuned_effort_uadfv.pth \
#     --image /Volumes/Crucial/Large_Downloads/AI/DATASETS/DFB/rgb/UADFV

# =============================================
# 5. ADVANCED FINE-TUNING WITH CUSTOM PARAMETERS
# =============================================
# uv run DeepfakeBench/training/finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --train_dataset UADFV \
#     --test_dataset UADFV \
#     --pretrained_weights /path/to/pretrained.pth \
#     --train_batchSize 4 \
#     --nEpochs 15

# =============================================
# USAGE NOTES
# =============================================
# - Uncomment the specific command you want to run
# - The fine-tuning configuration uses optimized parameters for SVD-based fine-tuning
# - See FINETUNING_README.md for detailed usage instructions
# - Fine-tuning leverages the SVD decomposition approach for efficient parameter updates
# - Only ~1-5% of model parameters are trainable during fine-tuning