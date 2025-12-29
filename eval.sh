# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
	--detector_config \
		'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
	--weights \
		'DeepfakeBench/training/logs/batchFacesAll/test/combined_test/batchFacesAll.pth' \
	--test_dataset \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/CollabDiff' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/deepfacelab' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/heygen_new' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/MidJourney' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/stargan' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/starganv2' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/styleclip' \
		'/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/whichfaceisreal' \
	--output_dir \
		'evaluation_results'
