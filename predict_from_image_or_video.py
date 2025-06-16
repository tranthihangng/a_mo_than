import cv2
import numpy as np
import joblib
import sys
from feature_extractor import CoalFeatureExtractor

def load_model_and_scaler():
    model = joblib.load('coal_model.pkl')
    scaler = joblib.load('scaler.pkl')
    try:
        le = joblib.load('label_encoder.pkl')
    except:
        le = None
    return model, scaler, le

def extract_features_from_image(image, roi_points=None):
    extractor = CoalFeatureExtractor()
    if roi_points is not None:
        mask = np.zeros(image.shape, dtype=np.uint8)
        points = roi_points.astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        mask = np.ones_like(image) * 255
    features = extractor.extract_ml_features(image, mask)
    vector = extractor.get_feature_vector(features)
    return vector

def predict_image(image_path, model, scaler, le, roi_points=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    feature_vector = extract_features_from_image(image, roi_points)
    feature_vector = scaler.transform([feature_vector])
    pred = model.predict(feature_vector)
    proba = model.predict_proba(feature_vector)[0]
    if le:
        pred = le.inverse_transform(pred)
        class_names = le.inverse_transform(np.arange(len(proba)))
    else:
        class_names = np.arange(len(proba))
    print(f"Ảnh {image_path} => Dự đoán: {pred[0]}")
    print("Xác suất từng class:")
    for cls, p in zip(class_names, proba):
        print(f"  {cls}: {p:.4f}")
    return pred[0], proba

def predict_video(video_path, model, scaler, le, roi_points=None, step=10):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feature_vector = extract_features_from_image(gray, roi_points)
            feature_vector = scaler.transform([feature_vector])
            pred = model.predict(feature_vector)
            proba = model.predict_proba(feature_vector)[0]
            if le:
                pred = le.inverse_transform(pred)
                class_names = le.inverse_transform(np.arange(len(proba)))
            else:
                class_names = np.arange(len(proba))
            print(f"Frame {frame_idx}: Dự đoán mức than: {pred[0]}")
            print("  Xác suất từng class:")
            for cls, p in zip(class_names, proba):
                print(f"    {cls}: {p:.4f}")
        frame_idx += 1
    cap.release()

def main():
    # Định nghĩa ROI nếu cần
    roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])
    model, scaler, le = load_model_and_scaler()

    # Đổi đường dẫn dưới đây cho phù hợp
    #input_path = "F:\phan_loai\full\hiv00007_000109.jpg"  # Ví dụ: "test.jpg" hoặc "test.mp4"
    input_path = r"F:\phan_loai\empty\hiv00016_000847.jpg"
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        predict_image(input_path, model, scaler, le, roi_points)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        predict_video(input_path, model, scaler, le, roi_points, step=10)
    else:
        print("Định dạng file không hỗ trợ!")

if __name__ == "__main__":
    main()
