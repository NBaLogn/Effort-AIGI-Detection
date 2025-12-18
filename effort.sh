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
# uv run DeepfakeBench/training/finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --train_dataset UADFV \
#     --test_dataset UADFV \
#     --pretrained_weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort_clip_L14_trainOn_FaceForensic.pth

#use direct image loading instead of the processed images
#use direct paths
# uv run DeepfakeBench/training/finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --train_dataset "/Volumes/Crucial/Large_Downloads/AI/DATASETS/Chameleon_retinafaces" \
#     --test_dataset "/Volumes/Crucial/Large_Downloads/SAMPLED/GenImage_sampled2/val" \
#     --pretrained_weights "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/logs/ff_12-16-15-57-38/test/avg/ckpt_best.pth"

# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================
uv run DeepfakeBench/training/evaluate_finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --weights "/Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort/effort_clip_L14_trainOn_FaceForensic.pth" \
    --test_dataset "/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_faceswap1000" \
    --output_dir evaluation_results
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/deepfacelab_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/heygen_new_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/MidJourney_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/stargan_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/starganv2_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/styleclip_sampled" \
    # "/Volumes/Crucial/Large_Downloads/SAMPLED/df40/whichfaceisreal_sampled" \

# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================
# uv run DeepfakeBench/training/perf_final.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat \
#     --weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort/effort_clip_L14_trainOn_chameleon.pth \
#     --image /Volumes/Crucial/Large_Downloads/AI/DATASETS/Chameleon

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