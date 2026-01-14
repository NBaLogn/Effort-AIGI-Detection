# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/inference.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights \
        'DeepfakeBench/training/weights/finetuned/newBatchFaces.pth' \
    --image \
        '/Volumes/Crucial/AI/DATASETS/SAMPLED/Genimage_faces_09_flat' \
    --limit 0 \
    # --landmark_model \
        # 'DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat' \