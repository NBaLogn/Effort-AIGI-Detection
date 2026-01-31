# =============================================
# 2. FINE-TUNING 
# =============================================
# use direct image loading instead of the processed images
# 

#newBatchAll2
# train on low AUC sets, test on high AUC sets
uv run DeepfakeBench/training/finetune.py \
    --detector_config \
        DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_EchoMimic \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_DaGAN \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_FSRT \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_LIA \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_LivePortrait \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_MCNET \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_TPSMM \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_BlendFace \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_Celeb-DF-v2 \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_GHOST \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_HifiFace \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_InSwapper \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_MobileFaceSwap \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_SimSwap \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceSwap_UniFace \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_AniTalker \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_IP_LAP \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_SadTalker \
        \
        DeepfakeBench/training/facedata/Chameleon_retinafaces_even_cropped \
        \
        DeepfakeBench/training/facedata/deepfake_in_the_wild_22k_even_cropped \
        \
        DeepfakeBench/training/facedata/df40_even_cropped/deepfacelab \
        DeepfakeBench/training/facedata/df40_even_cropped/heygen_new \
        \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/I2I_DreamBooth \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/T2I_HPS \
        \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DDIM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/Inpaint \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/PNDM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p3g7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p4g5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p5g3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p0g5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p1g7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p2g3 \
        \
        DeepfakeBench/training/facedata/humans100k_even_cropped \
        \
        DeepfakeBench/training/facedata/quan_faceswap2000_even_cropped/ \
        \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/AniPortraitAudio \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/AniPortraitVideo \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/EmoPortrait \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/Hallo \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/Hallo2 \
        DeepfakeBench/training/facedata/U_TalkingHead_cropped/LivePortrait \
        \
    --test_dataset \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/FaceReenact_HyperReenact \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_EDTalk \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_FLOAT \
        DeepfakeBench/training/facedata/Celeb-DF-v3_even_cropped/TalkingFace_Real3DPortrait \
        \
        DeepfakeBench/training/facedata/df40_even_cropped/CollabDiff \
        DeepfakeBench/training/facedata/df40_even_cropped/stargan \
        DeepfakeBench/training/facedata/df40_even_cropped/starganv2 \
        DeepfakeBench/training/facedata/df40_even_cropped/styleclip \
        DeepfakeBench/training/facedata/df40_even_cropped/MidJourney \
        DeepfakeBench/training/facedata/df40_even_cropped/whichfaceisreal \
        \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/FE_CoDiff \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/FE_cycle_diff \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/FE_Imagic \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/FS_DCFace \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/FS_DiffFace \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/I2I_FreeDoM_I \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/I2I_LoRA \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/I2I_SDXL_Refine \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/T2I_FreeDoM_T \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/T2I_Midjourney \
        DeepfakeBench/training/facedata/DiFF_30k_even_cropped/T2I_SDXL \
        \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/ADM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DDPM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DiffSwap \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/LDM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/Wild \
        \
        DeepfakeBench/training/facedata/quan_dataset_even_cropped/ \
    --pretrained_weights \
        DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth
