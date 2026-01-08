# =============================================
# 2. FINE-TUNING 
# =============================================
# use direct image loading instead of the processed images
# 

#newBatchAll
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset \
        'DeepfakeBench/training/facedata/Chameleon_retinafaces/' \
        'DeepfakeBench/training/facedata/Genimage_faces/' \
        'DeepfakeBench/training/facedata/quan_dataset/' \
        'DeepfakeBench/training/facedata/quan_faceswap2000/' \
        'DeepfakeBench/training/facedata/ivansivkovenin_faces' \
    --test_dataset \
        'DeepfakeBench/training/facedata/df40/CollabDiff' \
        'DeepfakeBench/training/facedata/df40/deepfacelab' \
        'DeepfakeBench/training/facedata/df40/heygen_new' \
        'DeepfakeBench/training/facedata/df40/MidJourney' \
        'DeepfakeBench/training/facedata/df40/stargan' \
        'DeepfakeBench/training/facedata/df40/starganv2' \
        'DeepfakeBench/training/facedata/df40/styleclip' \
        'DeepfakeBench/training/facedata/df40/whichfaceisreal' \
    --pretrained_weights \
        'DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth'

# batchAll-2
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset \
        'DeepfakeBench/training/facedata/Chameleon_retinafaces/' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/ADM' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/BigGAN' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/glide' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/Midjourney' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_4' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_5' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/VQDM' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/wukong' \
        'DeepfakeBench/training/facedata/quan_dataset/' \
        'DeepfakeBench/training/facedata/quan_faceswap2000/' \
        'DeepfakeBench/training/facedata/ivansivkovenin_faces' \
    --test_dataset \
        'DeepfakeBench/training/facedata/df40/CollabDiff' \
        'DeepfakeBench/training/facedata/df40/deepfacelab' \
        'DeepfakeBench/training/facedata/df40/heygen_new' \
        'DeepfakeBench/training/facedata/df40/MidJourney' \
        'DeepfakeBench/training/facedata/df40/stargan' \
        'DeepfakeBench/training/facedata/df40/starganv2' \
        'DeepfakeBench/training/facedata/df40/styleclip' \
        'DeepfakeBench/training/facedata/df40/whichfaceisreal' \
    --pretrained_weights \
        'DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth'