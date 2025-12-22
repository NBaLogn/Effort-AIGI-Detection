# =============================================
# 2. FINE-TUNING 
# =============================================

#use direct image loading instead of the processed images
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