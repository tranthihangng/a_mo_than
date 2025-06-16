import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('F:\phan_loai\T\hiv00008_000220.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tọa độ hai polygon
points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]], np.int32)
points_hhook = np.array([[911, 957], [1423, 647], [1629, 637], [1223,1013]], np.int32)
# Hoán đổi x ↔ y
#points_hhook = points_hhook[:, [1, 0]]  # hoặc dùng .copy() nếu muốn an toàn
# Vẽ polygon đầu tiên (màu xanh lá)
cv2.polylines(image_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=4)

# Vẽ polygon thứ hai (màu đỏ)
cv2.polylines(image_rgb, [points_hhook], isClosed=True, color=(255, 0, 0), thickness=4)

# Tính trọng tâm của points_hhook
M = cv2.moments(points_hhook)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    # Vẽ trọng tâm bằng hình tròn lớn và dấu '+'
    cv2.drawMarker(image_rgb, (cx, cy), color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=5)
    cv2.circle(image_rgb, (cx, cy), 20, (255, 0, 255), -1)
else:
    print("Không tính được trọng tâm (diện tích polygon = 0)")

# Lưu ảnh kết quả
cv2.imwrite('output_roi_hook.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

# Hiển thị kết quả
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.title("Polygon ROI & Hook ROI with Centroid")
plt.axis('off')
plt.show()
