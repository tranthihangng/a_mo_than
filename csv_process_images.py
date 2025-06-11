import os
import cv2
import numpy as np
import pandas as pd
from feature_extractor import CoalFeatureExtractor

def process_image_folder(folder_path, output_csv, roi_points=None):
    """
    Xử lý tất cả ảnh trong folder và lưu đặc trưng vào CSV
    
    Args:
        folder_path (str): Đường dẫn đến folder chứa ảnh
        output_csv (str): Đường dẫn để lưu file CSV
        roi_points (np.array, optional): Mảng các điểm polygon xác định vùng ROI
    """
    # Khởi tạo feature extractor
    extractor = CoalFeatureExtractor()
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Danh sách để lưu features của từng ảnh
    all_features = []
    
    print(f"Bắt đầu xử lý {len(image_files)} ảnh...")
    
    # Xử lý từng ảnh
    for i, image_file in enumerate(image_files, 1):
        try:
            # Đọc ảnh
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Không thể đọc ảnh: {image_file}")
                continue
            
            # Tạo mask ROI
            if roi_points is not None:
                roi_mask = np.zeros(image.shape, dtype=np.uint8)
                points = roi_points.astype(np.int32)
                cv2.fillPoly(roi_mask, [points], 255)
            else:
                roi_mask = np.ones_like(image) * 255
            
            # Trích xuất đặc trưng
            features = extractor.extract_ml_features(image, roi_mask)
            all_features.append(features)
            
            print(f"Đã xử lý ảnh {i}/{len(image_files)}: {image_file}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_file}: {str(e)}")
    
    # Tạo DataFrame và lưu vào CSV
    if all_features:
        df = extractor.create_feature_dataframe(all_features, labels=["full"] * len(all_features))
        df.to_csv(output_csv, index=False)
        print(f"\nĐã lưu {len(df)} mẫu vào file {output_csv}")
    else:
        print("Không có ảnh nào được xử lý thành công")

def main():
    # Đường dẫn folder chứa ảnh
    #folder_path = "F:\phan_loai\full"  # Thư mục hiện tại
    folder_path = r"F:\phan_loai\full"

    
    # Đường dẫn file CSV đầu ra
    output_csv = "features_full.csv"
    
    # Định nghĩa các điểm polygon ROI
    roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])
    
    # Xử lý folder ảnh
    process_image_folder(folder_path, output_csv, roi_points)

if __name__ == "__main__":
    main() 