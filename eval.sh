# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
	--detector_config \
		'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
	--weights \
		'DeepfakeBench/training/weights/finetuned/newBatchFaces.pth' \
	--test_dataset \
		'DeepfakeBench/training/facedata/Genimage_faces_09_flat' \
	--output_dir 'evaluation_results'
		# 'DeepfakeBench/training/facedata/Chameleon_retinafaces' \
		# 'DeepfakeBench/training/facedata/df40/CollabDiff' \
        # 'DeepfakeBench/training/facedata/df40/deepfacelab' \
        # 'DeepfakeBench/training/facedata/df40/heygen_new' \
        # 'DeepfakeBench/training/facedata/df40/MidJourney' \
        # 'DeepfakeBench/training/facedata/df40/stargan' \
        # 'DeepfakeBench/training/facedata/df40/starganv2' \
        # 'DeepfakeBench/training/facedata/df40/styleclip' \
        # 'DeepfakeBench/training/facedata/df40/whichfaceisreal' \
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
