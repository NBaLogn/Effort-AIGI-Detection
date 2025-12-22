# =============================================
# 4. INFERENCE WITH FINE-TUNED MODEL
# =============================================

uv run DeepfakeBench/training/inference.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat \
    --weights 'DeepfakeBench/training/logs/batchFaces2000/test/combined_test/batchFaces2000.pth' \
    --image '/Volumes/Crucial/Large_Downloads/AI/DATASETS/SAMPLED/quan_dataset_2000/val'