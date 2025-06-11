import cv2
import numpy as np
from feature_extractor import CoalFeatureExtractor

def create_polygon_mask(image_shape, points):
    """
    Tạo mask từ các điểm polygon
    
    Args:
        image_shape (tuple): Kích thước ảnh (height, width)
        points (np.array): Mảng các điểm polygon [[x1,y1], [x2,y2], ...]
    
    Returns:
        np.array: Mask nhị phân với vùng ROI là 255, nền là 0
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = points.astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def test_feature_extraction(image_path, roi_points=None):
    """
    Hàm test trích xuất đặc trưng từ một ảnh
    
    Args:
        image_path (str): Đường dẫn đến file ảnh
        roi_points (np.array, optional): Mảng các điểm polygon xác định vùng ROI
    
    Returns:
        dict: Dictionary chứa các đặc trưng được trích xuất
    """
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    # Tạo mask ROI
    if roi_points is not None:
        roi_mask = create_polygon_mask(image.shape, roi_points)
    else:
        # Nếu không có ROI, sử dụng toàn bộ ảnh
        roi_mask = np.ones_like(image) * 255
    
    # Khởi tạo feature extractor
    extractor = CoalFeatureExtractor()
    
    # Trích xuất đặc trưng
    features = extractor.extract_ml_features(image, roi_mask)
    
    # In thông tin về các đặc trưng
    print("\nCác đặc trưng được trích xuất:")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value:.4f}")
    
    return features

def main():
    # Ví dụ sử dụng với polygon ROI
    image_path = "hiv00007_000302.jpg"  # Thay đổi đường dẫn ảnh của bạn
    
    # Định nghĩa các điểm polygon ROI
    roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])
    
    try:
        features = test_feature_extraction(image_path, roi_points)
        print("\nTổng số đặc trưng:", len(features))
        
        # Tạo feature vector
        extractor = CoalFeatureExtractor()
        feature_vector = extractor.get_feature_vector(features)
        print("\nFeature vector shape:", feature_vector.shape)
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main() 