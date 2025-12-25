# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/evaluate_finetune.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights \
        'DeepfakeBench/training/logs/batchFacesAll/test/combined_test/ckpt_best.pth' \
    --test_dataset \
        'DeepfakeBench/facedata/quan_faceswap2000/val' \
        'DeepfakeBench/facedata/Chameleon_retinafaces/val' \
    --output_dir \
        'evaluation_results'
    