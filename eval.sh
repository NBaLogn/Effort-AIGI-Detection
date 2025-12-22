# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

# uv run DeepfakeBench/training/evaluate_finetune.py \
#     --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
#     --weights "DeepfakeBench/training/logs/batchFaces2000/test/combined_test/batchFaces2000.pth" \
#     --test_dataset "/Volumes/Crucial/Large_Downloads/AI/DATASETS/quan_dataset/val" \
#     --output_dir evaluation_results
    
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/heygen_new" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/MidJourney" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/stargan" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/starganv2" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/styleclip" \
    # "/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/df40/whichfaceisreal" \