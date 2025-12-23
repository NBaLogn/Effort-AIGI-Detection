# =============================================
# 2. FINE-TUNING 
# =============================================
# use direct image loading instead of the processed images

uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset 'DeepfakeBench/facedata/Chameleon_retinafaces/train' \
                    'DeepfakeBench/facedata/GenImageFaces/train' \
                    'DeepfakeBench/facedata/quan_dataset/train' \
                    'DeepfakeBench/facedata/quan_faceswap2000/train' \
    --test_dataset 'DeepfakeBench/facedata/Chameleon_retinafaces/val' \
                   'DeepfakeBench/facedata/GenImageFaces/val' \
                   'DeepfakeBench/facedata/quan_faceswap2000/val' \
                   'DeepfakeBench/facedata/quan_dataset/val' \
    --pretrained_weights '/Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort/effort_clip_L14_trainOn_FaceForensic.pth'