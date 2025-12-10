# train
# uv run /Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/mac_train.py --detector_path /Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml --train_dataset UADFV --test_dataset UADFV

# inference
# uv run DeepfakeBench/training/perf_final.py --detector_config DeepfakeBench/training/config/detector/effort.yaml --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat --weights /Volumes/Crucial/Large_Downloads/AI/WEIGHTS/effort_clip_L14_trainOn_FaceForensic.pth --image /Volumes/Crucial/Large_Downloads/AI/DATASETS/DFB/rgb/UADFV