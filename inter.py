import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

# 1. Đọc ảnh gốc
image = cv2.imread('hiv00007_000045.jpg')

# 2. Định nghĩa vùng polygon (ROI)
points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]], np.int32)

# 3. Tạo mask để lấy vùng ROI
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [points], 255)

# 4. Cắt vùng ROI từ ảnh gốc
roi = cv2.bitwise_and(image, image, mask=mask)

# 5. Chuyển ROI sang ảnh xám
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 6. Lấy tọa độ và giá trị điểm ảnh trong vùng mask
y, x = np.where(mask == 255)
z = gray_roi[y, x]

# 7. Nội suy thành mặt lưới đều
grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# 8. Tạo biểu đồ 3D và render ảnh từ góc nhìn mong muốn
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ bề mặt nội suy với colormap
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0, antialiased=False)

# Tùy chỉnh góc nhìn nếu cần
ax.view_init(elev=45, azim=135)  # Cao độ và góc xoay

# Tắt trục để có ảnh rõ ràng
ax.axis('off')

# Render ảnh từ khung vẽ matplotlib
fig.canvas.draw()
image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# Đóng figure để giải phóng bộ nhớ
plt.close(fig)

# 9. Resize ảnh nếu cần để làm input YOLO (ví dụ 416x416 hoặc 640x640)
image_resized = cv2.resize(image_from_plot, (416, 416))

# 10. Hiển thị ảnh kết quả
plt.imshow(image_resized)
plt.title("YOLO Input from 3D Surface View")
plt.axis('off')
plt.show()

# 11. Lưu ảnh nếu cần
cv2.imwrite('yolo_input_from_3d_view.png', cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
