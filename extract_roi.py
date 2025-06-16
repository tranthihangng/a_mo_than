import cv2
import numpy as np
import os
from pathlib import Path

def extract_roi_from_image(image_path, roi_points, output_path):
    """
    Extract ROI from a single image and save it
    Args:
        image_path: Path to input image
        roi_points: List of points defining the ROI polygon
        output_path: Path to save the extracted ROI
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return False

    # Convert roi_points to numpy array
    roi_points = np.array(roi_points, dtype=np.int32)

    # Create a mask for the ROI
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    # Create a black image with same size as original
    black_mask = np.zeros_like(image)
    
    # Apply the mask to get the ROI while keeping original size
    roi = cv2.bitwise_and(image, image, mask=mask)
    
    # Combine ROI with black mask for non-ROI areas
    result = cv2.add(roi, black_mask)

    # Save the result
    cv2.imwrite(str(output_path), result)
    return True

def process_folder(input_folder, output_folder, roi_points):
    """
    Process all images in a folder
    Args:
        input_folder: Path to input folder containing images
        output_folder: Path to save extracted ROIs
        roi_points: List of points defining the ROI polygon
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]

    # Process each image
    for image_path in image_files:
        output_path = Path(output_folder) / f"roi_{image_path.name}"
        if extract_roi_from_image(image_path, roi_points, output_path):
            print(f"Đã xử lý: {image_path.name}")

def main():
    # Định nghĩa các điểm polygon ROI (có thể thay đổi tùy theo nhu cầu)
    roi_points = [
        [663, 839],
        [1201, 661],
        [1717, 663],
        [1785, 1021]
    ]

    # ===== CẤU HÌNH ĐƯỜNG DẪN =====
    # Chọn chế độ xử lý: 1 cho một ảnh, 2 cho cả folder
    mode = 2

    if mode == 1:
        # Xử lý một ảnh
        image_path = r"F:F:\phan_loai\medium\hiv00007_000045.jpg"  # Đường dẫn ảnh input
        output_path = r"F:\coal_frame\roi_output\roi_image.jpg"      # Đường dẫn lưu ảnh ROI
        extract_roi_from_image(image_path, roi_points, output_path)
        print("Đã xử lý xong!")

    elif mode == 2:
        # Xử lý cả folder
        input_folder = r"F:\phan_loai\empty"                     # Folder chứa ảnh input
        output_folder = r"F:\output_roi\empty"                  # Folder lưu ảnh ROI
        process_folder(input_folder, output_folder, roi_points)
        print("Đã xử lý xong tất cả ảnh!")

    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 