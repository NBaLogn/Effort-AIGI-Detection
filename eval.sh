# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
	--detector_config \
		'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
	--weights \
		'DeepfakeBench/training/weights/finetuned/newBatchFaces.pth' \
	--test_dataset \
		'DeepfakeBench/training/facedata/Chameleon_retinafaces' \
		'DeepfakeBench/training/facedata/quan_dataset' \
		'DeepfakeBench/training/facedata/quan_faceswap2000'
	--output_dir 'evaluation_results'

		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/CollabDiff' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/deepfacelab' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/heygen_new' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/MidJourney' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/stargan' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/starganv2' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/styleclip' \
		# '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/whichfaceisreal' \