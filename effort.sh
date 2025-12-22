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
# 2. FINE-TUNING 
# =============================================
# uv run DeepfakeBench/training/finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --train_dataset UADFV \
#     --test_dataset UADFV \
#     --pretrained_weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort_clip_L14_trainOn_FaceForensic.pth

#use direct image loading instead of the processed images
#use direct paths
# uv run '/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/finetune.py' \
#     --detector_config '/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort_finetune.yaml' \
#     --train_dataset '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/Chameleon_retinafaces/train' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/GenImageFaces_2ndpass/train' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/quan_faceswap2000/train' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_dataset/train' \
#     --test_dataset '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/Chameleon_retinafaces/val' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/GenImageFaces_2ndpass/val' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/quan_faceswap2000/val' \
#     '/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_dataset/val' \
#     --pretrained_weights '/Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort/effort_clip_L14_trainOn_FaceForensic.pth'

# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================
# uv run DeepfakeBench/training/evaluate_finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --weights "DeepfakeBench/training/logs/batchFaces2000/test/combined_test/batchFaces2000.pth" \
#     --test_dataset "/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_dataset/val" \
#     --output_dir evaluation_results
    
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/heygen_new" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/MidJourney" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/stargan" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/starganv2" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/styleclip" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/whichfaceisreal" \

# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================
uv run DeepfakeBench/training/inference.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat \
    --weights 'DeepfakeBench/training/logs/batchFaces2000/test/combined_test/batchFaces2000.pth' \
    --image '/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_dataset/val'

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