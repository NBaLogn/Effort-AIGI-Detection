# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights \
        'DeepfakeBench/training/logs/batchFaces2000/test/combined_test/batchFaces2000.pth' \
    --test_dataset \
        'DeepfakeBench/facedata/quan_faceswap2000/train' \
    --output_dir \
        'evaluation_results'
    