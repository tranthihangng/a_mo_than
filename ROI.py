# import cv2
# import numpy as np

# # Đường dẫn ảnh
# image_path = r"hiv00007_000302.jpg"  # ← Đổi tên ảnh tương ứng

# # Đọc ảnh
# image = cv2.imread(image_path)

# # Kiểm tra ảnh có đọc được không
# if image is None:
#     print("Không thể đọc ảnh.")
#     exit()

# # Định nghĩa các điểm polygon ROI
# roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])

# # Vẽ đường viền ROI
# image_with_roi = image.copy()
# cv2.polylines(image_with_roi, [roi_points], isClosed=True, color=(0, 255, 0), thickness=3)

# # Hiển thị ảnh
# cv2.imshow("Image with ROI", image_with_roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#scale
import cv2
import numpy as np

# Đường dẫn ảnh
image_path = r"F:\coal_frame\hiv00007\hiv00007_000001.jpg"  # ← Đổi tên ảnh tương ứng

# Đọc ảnh
image = cv2.imread(image_path)

# Kiểm tra ảnh có đọc được không
if image is None:
    print("Không thể đọc ảnh.")
    exit()

# Định nghĩa các điểm polygon ROI
roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])

# Resize ảnh để dễ hiển thị (ví dụ resize theo chiều rộng 1000px)
scale_percent = 50  # Tỉ lệ phần trăm so với kích thước gốc
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize ảnh và điểm ROI tương ứng
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
resized_roi_points = (roi_points * scale_percent / 100).astype(int)

# Vẽ đường viền ROI trên ảnh đã resize
image_with_roi = resized_image.copy()
cv2.polylines(image_with_roi, [resized_roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

# Hiển thị ảnh
cv2.imshow("Image with ROI", image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
