# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
	--detector_config \
		'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
	--weights \
		'DeepfakeBench/training/weights/finetuned/newBatchFaces.pth' \
	--test_dataset \
		'/Volumes/Crucial/AI/DATASETS/df40/_MIXED/CollabDiff' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/deepfacelab' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/heygen_new' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/MidJourney' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/stargan' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/starganv2' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/styleclip' \
        '/Volumes/Crucial/AI/DATASETS/df40/_MIXED/whichfaceisreal' \
	--output_dir 'evaluation_results'

		# 'DeepfakeBench/training/facedata/TalkingHead_processed/AniPortraitAudio' \
		# 'DeepfakeBench/training/facedata/TalkingHead_processed/AniPortraitVideo' \
		# 'DeepfakeBench/training/facedata/TalkingHead_processed/EmoPortrait' \
		# 'DeepfakeBench/training/facedata/TalkingHead_processed/Hallo' \
		# 'DeepfakeBench/training/facedata/TalkingHead_processed/Hallo2' \
		# 'DeepfakeBench/training/facedata/TalkingHead_processed/LivePortrait' \

		# '/Volumes/Crucial/AI/DATASETS/OpenDataLab___ForgeryNet' \

		# '/Volumes/Crucial/AI/DATASETS/humans100k_jpg' \

		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/FE_CoDiff' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/FE_Imagic' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/FE_cycle_diff' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/FS_DCFace' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/FS_DiffFace' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/I2I_DreamBooth' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/I2I_FreeDoM_I' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/I2I_LoRA' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/I2I_SDXL_Refine' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/T2I_FreeDoM_T' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/T2I_HPS' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/T2I_Midjourney' \
		# '/Volumes/Crucial/AI/DATASETS/DiFF/DATA/T2I_SDXL' \
		
		# '/Volumes/Crucial/AI/DATASETS/TalkingHead_processed_additional_dataset_optimized/Hallo3' \
		# '/Volumes/Crucial/AI/DATASETS/TalkingHead_processed_additional_dataset_optimized/MAGI-1' \

		# 'DeepfakeBench/training/facedata/df40/CollabDiff' \
        # 'DeepfakeBench/training/facedata/df40/deepfacelab' \
        # 'DeepfakeBench/training/facedata/df40/heygen_new' \
        # 'DeepfakeBench/training/facedata/df40/MidJourney' \
        # 'DeepfakeBench/training/facedata/df40/stargan' \
        # 'DeepfakeBench/training/facedata/df40/starganv2' \
        # 'DeepfakeBench/training/facedata/df40/styleclip' \
        # 'DeepfakeBench/training/facedata/df40/whichfaceisreal' \

		# 'DeepfakeBench/training/facedata/Chameleon_retinafaces' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/ADM' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/BigGAN' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/glide' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/Midjourney' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_4' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/stable_diffusion_v_1_5' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/VQDM' \
		# 'DeepfakeBench/training/facedata/GenImage_faces_09/wukong' \

		# 'DeepfakeBench/training/facedata/quan_dataset' \
		# 'DeepfakeBench/training/facedata/quan_faceswap2000' \
