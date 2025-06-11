# HƯỚNG DẪN SỬ DỤNG PIPELINE PHÂN LOẠI MỨC THAN

## 1. Trích xuất đặc trưng (feature extraction)

**File:** `feature_extractor.py`

- Dùng class `CoalFeatureExtractor` để trích xuất đặc trưng từ ảnh hoặc vùng ROI của ảnh.
- Có thể sử dụng trực tiếp hoặc thông qua script ví dụ như `process_images.py` để trích xuất đặc trưng cho nhiều ảnh và lưu ra file CSV.

**Ví dụ sử dụng:**
```python
from feature_extractor import CoalFeatureExtractor
import cv2
import numpy as np

image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
roi_mask = np.ones_like(image) * 255  # hoặc tạo mask theo polygon
extractor = CoalFeatureExtractor()
features = extractor.extract_ml_features(image, roi_mask)
```

## 2. Huấn luyện model (train model)

**File:** `train_model.py`

- Đọc dữ liệu đặc trưng từ file CSV (ví dụ: `merged_data_finalxyyz.csv`).
- Cột cuối cùng phải là nhãn (class).
- Script sẽ tự động encode nhãn, chuẩn hóa đặc trưng, train model RandomForest và lưu lại 3 file trọng số:
  - `coal_model.pkl`: file model đã huấn luyện
  - `scaler.pkl`: bộ chuẩn hóa đặc trưng
  - `label_encoder.pkl`: bộ mã hóa nhãn (nếu nhãn là dạng text)

**Chạy lệnh:**
```bash
python train_model.py
```

## 3. Dự đoán kết quả (predict)

**File:** `predict_from_image_or_video.py`

- Dùng để dự đoán mức than cho ảnh hoặc video.
- Script sẽ tự động trích xuất đặc trưng, chuẩn hóa, và dự đoán bằng model đã train.
- In ra nhãn dự đoán và xác suất cho từng class.
- Có thể chỉnh sửa đường dẫn ảnh/video và ROI trong file.

**Chạy lệnh:**
```bash
python predict_from_image_or_video.py
```

## 4. Quy trình tổng quát

1. **Trích xuất đặc trưng** cho tất cả ảnh, lưu ra file CSV.
2. **Huấn luyện model** với file CSV đặc trưng, sinh ra các file trọng số.
3. **Dự đoán** cho ảnh hoặc video mới bằng script predict, sử dụng các file trọng số đã lưu.

---

**Yêu cầu thư viện:**
- numpy
- pandas
- scikit-learn
- opencv-python
- scikit-image
- joblib

Cài đặt nhanh:
```bash
pip install numpy pandas scikit-learn opencv-python scikit-image joblib
```

---

**Mọi thắc mắc hoặc cần hỗ trợ, vui lòng liên hệ tác giả code.** 