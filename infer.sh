# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================

uv run 'DeepfakeBench/training/inference.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --landmark_model \
        'DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat' \
    --weights \
        'DeepfakeBench/training/weights/batchFacesAll.pth' \
    --image \
        'DeepfakeBench/training/facedata/quan_faceswap2000/val'