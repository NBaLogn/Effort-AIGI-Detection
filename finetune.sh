# =============================================
# 2. FINE-TUNING 
# =============================================
# use direct image loading instead of the processed images
# 

#newBatchAll2
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset \
        'DeepfakeBench/training/facedata/Chameleon_retinafaces/' \
        \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/FE_CoDiff' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/FE_cycle_diff' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/FE_Imagic' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/FS_DCFace' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/FS_DiffFace' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/I2I_DreamBooth' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/I2I_FreeDoM_I' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/I2I_LoRA' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/I2I_SDXL_Refine' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/T2I_FreeDoM_T' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/T2I_HPS' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/T2I_Midjourney' \
        'DeepfakeBench/training/facedata/DiFF_sampled_30k/T2I_SDXL' \
        \
		'DeepfakeBench/training/facedata/GenImage_faces_09/ADM' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/BigGAN' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/glide' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/Midjourney' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_4' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_5' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/VQDM' \
		'DeepfakeBench/training/facedata/GenImage_faces_09/wukong' \
        \
        'DeepfakeBench/training/facedata/humans100k_jpg' \
        'DeepfakeBench/training/facedata/quan_dataset/' \
        'DeepfakeBench/training/facedata/quan_faceswap2000/' \
        \
        'DeepfakeBench/training/facedata/TalkingHead_processed/AniPortraitAudio' \
        'DeepfakeBench/training/facedata/TalkingHead_processed/AniPortraitVideo' \
        'DeepfakeBench/training/facedata/TalkingHead_processed/EmoPortrait' \
        'DeepfakeBench/training/facedata/TalkingHead_processed/Hallo' \
        'DeepfakeBench/training/facedata/TalkingHead_processed/Hallo2' \
        'DeepfakeBench/training/facedata/TalkingHead_processed/LivePortrait' \
    --test_dataset \
        'DeepfakeBench/training/facedata/df40_processed/CollabDiff' \
        'DeepfakeBench/training/facedata/df40_processed/deepfacelab' \
        'DeepfakeBench/training/facedata/df40_processed/heygen_new' \
        'DeepfakeBench/training/facedata/df40_processed/MidJourney' \
        'DeepfakeBench/training/facedata/df40_processed/stargan' \
        'DeepfakeBench/training/facedata/df40_processed/starganv2' \
        'DeepfakeBench/training/facedata/df40_processed/styleclip' \
        'DeepfakeBench/training/facedata/df40_processed/whichfaceisreal' \
    --pretrained_weights \
        'DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth'
