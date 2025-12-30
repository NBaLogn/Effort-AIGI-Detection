# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/inference.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --landmark_model \
        'DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat' \
    --weights \
        'DeepfakeBench/training/finetuned_weights/batchFacesAll.pth' \
    --image \
        'DeepfakeBench/training/facedata/ivansivkovenin_faces' \
    --limit 0