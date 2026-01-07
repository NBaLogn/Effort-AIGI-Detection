# 🎯 Tinh chỉnh mô hình Effort

Hướng dẫn này cung cấp các bước đầy đủ để tinh chỉnh mô hình Effort bằng phương pháp phân tích SVD.

## 🚀 Bắt đầu nhanh

### Các lệnh

### finetune.sh

```bash
uv run 'DeepfakeBench/training/finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --train_dataset '[DATASET_PATH]' \
    --test_dataset '[DATASET_PATH]' \
    --pretrained_weights '[PATH_TO]/effort_clip_L14_trainOn_FaceForensic.pth'
```

### eval.sh

```bash
uv run 'DeepfakeBench/training/evaluate_finetune.py' \
    --detector_config 'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --weights '[PATH_TO_FINETUNED_WEIGHT]'\
    --test_dataset '[DATASET_PATH]' '[DATASET_PATH]' \
    --output_dir '[PATH_TO_OUTPUT_FOLDER]'
```

### infer.sh

```bash
uv run 'DeepfakeBench/training/inference.py' \
    --detector_config \
        'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
    --landmark_model \
        '[PATH_TO]/shape_predictor_81_face_landmarks.dat' \
    --weights \
        '[PATH_TO_FINETUNED_WEIGHT]' \
    --image \
        '[PATH_TO_IMAGE_FILE_OR_FOLDER]'
```

## 📋 Cấu hình tinh chỉnh

Tệp cấu hình tinh chỉnh (`effort_finetune.yaml`) chứa các chế độ tối ưu như sau:

### Cấu hình dành cho batch2k

Sử dụng `effort_clip_L14_trainOn_FaceForensic.pth`, dùng 2000 khuôn mặt trích xuất từ các tập Chameleon, Genimage, quan và quanFaceSwap, tinh chỉnh trong 2 epoch. 

`--train_dataset` và `--test_dataset` giống nhau.

### Cấu hình dành cho batchAll

Sử dụng `effort_clip_L14_trainOn_FaceForensic.pth`, dùng tất cả khuôn mặt trích xuất từ các tập Chameleon, Genimage, quan và quanFaceSwap, tinh chỉnh trong 10 epoch. 

`--train_dataset` là các phần train, `--test_dataset` nhắm tới các phần val.

##### Cấu hình tinh chỉnh

```yaml
# Các tùy chọn riêng cho tinh chỉnh
fine_tune: true
pretrained_checkpoint: null
freeze_backbone: true
train_classification_head: true
train_svd_residuals: true

# Cấu hình huấn luyện
nEpochs: 10
lr_scheduler: cosine
lr_T_max: 10
lr_eta_min: 0.000001

# Cấu hình bộ tối ưu
optimizer:
  type: adam
  adam:
    lr: 0.00005
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001

# Tăng cường dữ liệu
data_aug:
  flip_prob: 0.5
  rotate_prob: 0.4
  rotate_limit: [-10, 10]
  blur_prob: 0.3
  brightness_prob: 0.3
  brightness_limit: [-0.1, 0.1]
```

### Cấu hình dành cho newBatch

Thêm tập Midjourney từ `ivansivkovenin`, dùng `cosine` cho `lr_scheduler`, bật `early stopping` khi AUC không cải thiện trên 0.0001 trong 2 epoch. Giảm một nửa tốc độ học.

`--train_dataset` bao gồm Chameleon, Genimage, tập của `ivansivkovenin`, quan và quanFS; `--test_dataset` nhắm vào df40. Tinh chỉnh dừng sau 5 epoch.

##### Cấu hình tinh chỉnh
```yaml
# Các công tắc tinh chỉnh (khớp với `effort_finetune.yaml`)
fine_tune: true
freeze_backbone: true
train_classification_head: true
train_svd_residuals: true
save_avg: true

# Lịch huấn luyện được điều chỉnh cho tập dữ liệu mở rộng
nEpochs: 5
lr_scheduler: cosine
lr_T_max: 5
lr_eta_min: 0.000001

# Cấu hình bộ tối ưu
optimizer:
  type: adam
  adam:
    lr: 0.000025   # giảm một nửa so với batchAll để ổn định tập dữ liệu lớn hơn
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001

early_stopping:
  enabled: true
  patience: 2
  min_delta: 0.0001   # dừng khi AUC không cải thiện nghĩa trong hai epoch
  metric: auc
```

## 🔧 Cách hoạt động của tinh chỉnh

### Phương pháp phân tích SVD

Mô hình Effort dùng **Phân tích không gian con trực giao** để tinh chỉnh hiệu quả:

1. **Ma trận trọng số gốc**: `W = U @ Σ @ Vᵀ`
2. **Các thành phần chính cố định**: `W_main = U_r @ Σ_r @ V_rᵀ` (r thành phần đầu)
3. **Phần dư có thể huấn luyện**: `W_residual = U_residual @ Σ_residual @ V_residualᵀ` (các thành phần còn lại)
4. **Tổng trọng số**: `W_total = W_main + W_residual`

### Hiệu suất tham số

- **Tham số cố định**: ~99% tổng tham số (giữ lại kiến thức tiền huấn luyện)
- **Tham số huấn luyện được**: ~1% tổng tham số (phần dư SVD + đầu phân loại)
- **Tổng tham số huấn luyện**: ~1-5% tham số mô hình

## 📊 Các chỉ số đánh giá

Script đánh giá cung cấp các chỉ số toàn diện:

- **Chỉ số chính**: AUC, EER, Accuracy, AP
- **Chỉ số bổ sung**: Precision, Recall, F1 Score
- **Ghi chép chi tiết**: tiến trình theo từng batch, tổng kết cuối cùng
- **Định dạng kết quả**: JSON để phân tích dễ dàng

## 🔍 Giám sát và gỡ lỗi

### Ghi log

- **Log tinh chỉnh**: `training/logs/finetuning.log`
- **Log đánh giá**: `evaluation_results/evaluation.log`
- **TensorBoard**: tự động ghi chỉ số

## 🛠️ Cài đặt

### 1. Clone kho mã
```bash
git clone https://github.com/your-repo/effort-aigi-detection.git
cd effort-aigi-detection
```

### 2. Thiết lập môi trường Python
```bash
# Cài phụ thuộc Python bằng uv
uv sync
```

Lệnh này sẽ cài mọi phụ thuộc Python trong `pyproject.toml`, bao gồm:
- FastAPI và Uvicorn cho backend
- PyTorch và các thư viện ML liên quan
- OpenCV, dlib và các công cụ thị giác máy tính khác
- Các phụ thuộc cho mô hình phát hiện deepfake

### 3. Thiết lập frontend
```bash
cd frontend
npm install
# hoặc
uv run npm install
```

Lệnh này sẽ cài các phụ thuộc Next.js và React.

### 4. Tải các mô hình cần thiết
Ứng dụng yêu cầu một số tệp mô hình cụ thể:

#### Mô hình phát hiện điểm mốc
Tải bộ dự đoán hình dạng khuôn mặt 81 điểm tại https://github.com/codeniko/shape_predictor_81_face_landmarks

#### Trọng số phát hiện deepfake
Bạn cần trọng số mô hình Effort đã được huấn luyện trước. Đặt ở vị trí phù hợp. Tệp `server.py` sẽ tìm cả trọng số và mô hình điểm mốc, bạn phải cập nhật đường dẫn nếu cần.

## 🚀 Chạy ứng dụng

### Chế độ phát triển

#### 1. Khởi động backend
```bash
# Từ thư mục gốc
uv run backend/server.py
```

Backend khởi động tại `http://0.0.0.0:8000` với:
- FastAPI REST API cho việc phát hiện deepfake
- CORS được bật để giao tiếp với frontend
- Tự động nạp mô hình và trực quan Grad-CAM
- Endpoint kiểm tra sức khỏe tại `/health`

#### 2. Khởi động frontend
```bash
cd frontend
npm run dev
# hoặc
uv run npm run dev
```

Frontend khởi động tại `http://localhost:3000` với:
- Hot module replacement để cập nhật nhanh
- Giao diện tương tác cho phát hiện deepfake
- Tải ảnh và phân tích trực tuyến
- Hiển thị Grad-CAM giải thích vùng ảnh

## 📊 API endpoints

### POST /predict
Tải ảnh lên để kiểm tra deepfake:

**Yêu cầu:**
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

**Phản hồi:**
```json
{
  "label": "FAKE",
  "score": 0.95,
  "reasoning": "Suspicious textures detected around the eyes",
  "grad_cam_image": "data:image/jpeg;base64,..."
}
```

### GET /health
Kiểm tra backend đang chạy:
```bash
curl http://localhost:8000/health
```

## 🎯 Cách sử dụng

1. **Tải ảnh lên**: Kéo thả hoặc chọn bằng trình duyệt file
2. **Xem kết quả**: Nhận kết quả phát hiện deepfake tức thời
3. **Phân tích heatmap**: Grad-CAM hiển thị vùng ảnh khiến mô hình quyết định
4. **Xử lý hàng loạt**: Tải nhiều ảnh cùng lúc để kiểm tra nhanh

## 📁 Cấu trúc dự án

```
effort-aigi-detection/
├── backend/                  # FastAPI backend
│   ├── server.py             # Ứng dụng backend chính
│   └── gradcam_utils.py      # Tiện ích Grad-CAM
├── frontend/                 # Frontend Next.js
│   ├── app/                  # Các trang ứng dụng
│   ├── components/           # Thành phần React
│   └── public/               # Tài nguyên tĩnh
├── DeepfakeBench/            # Logic phát hiện lõi
│   ├── training/             # Script huấn luyện
│   └── preprocessing/        # Xử lý dữ liệu
└── README.md                 # Tệp README gốc
```
