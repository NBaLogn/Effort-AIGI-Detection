# =============================================
# 2. FINE-TUNING 
# =============================================
# use direct image loading instead of the processed images

        # 'DeepfakeBench/training/facedata/Chameleon_retinafaces/' \
        # 'DeepfakeBench/training/facedata/Genimage_faces/' \
        # 'DeepfakeBench/training/facedata/quan_dataset/' \
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset \
        'DeepfakeBench/training/facedata/quan_faceswap2000/' \
        'DeepfakeBench/training/facedata/ivansivkovenin_faces' \
    --test_dataset \
        'DeepfakeBench/training/facedata/df40/CollabDiff' \
        'DeepfakeBench/training/facedata/df40/deepfacelab' \
    --pretrained_weights \
        'DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth'
        # 'DeepfakeBench/training/facedata/df40/heygen_new' \
        # 'DeepfakeBench/training/facedata/df40/MidJourney' \
        # 'DeepfakeBench/training/facedata/df40/stargan' \
        # 'DeepfakeBench/training/facedata/df40/starganv2' \
        # 'DeepfakeBench/training/facedata/df40/styleclip' \
        # 'DeepfakeBench/training/facedata/df40/whichfaceisreal' \