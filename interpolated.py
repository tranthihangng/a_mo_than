import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

# Đọc ảnh đầu vào
image = cv2.imread('hiv00007_000045.jpg')

# Tọa độ các điểm xác định vùng cần lấy (theo thứ tự: [x, y])
points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]], np.int32)

# Tạo mask để chọn phần ảnh bên trong polygon
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [points], 255)

# Cắt vùng polygon
roi = cv2.bitwise_and(image, image, mask=mask)

# Chuyển sang ảnh xám
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Lấy tọa độ pixel trong vùng ROI
y, x = np.where(mask == 255)
z = gray_roi[y, x]

# Tạo lưới đều và nội suy
grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# Vẽ các hình trên cùng một khung
fig = plt.figure(figsize=(20, 10))

# Ảnh gốc
ax1 = fig.add_subplot(221)
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')

# ROI
ax2 = fig.add_subplot(222)
ax2.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
ax2.set_title('Region of Interest (ROI)')
ax2.axis('off')

# Scatter 3D
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(x, y, z, c=z, cmap='viridis', marker='o')
ax3.set_title('3D Scatter from ROI')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Gray Value')

# Surface nội suy
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0, antialiased=False)
ax4.set_title('Interpolated 3D Surface')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Interpolated Value')

plt.tight_layout()
plt.show()

# Chuyển surface nội suy thành ảnh 2D (grayscale)


# Nếu muốn lưu lại ảnh 2D:
# cv2.imwrite('interpolated_2d.png', grid_z_img)

# --- Render biểu đồ 3D surface như ảnh 2D RGB ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0, antialiased=False)

# Tắt trục và khung
ax.set_axis_off()
plt.axis('off')

# Góc nhìn (có thể chỉnh lại cho phù hợp)
ax.view_init(elev=45, azim=135)  # elev: cao độ, azim: phương ngang

plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Lưu hình ảnh đúng như hiển thị
plt.savefig('surface_as_image.png', bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()

# Đọc lại ảnh RGB để dùng làm input YOLO
surface_img = cv2.imread('surface_as_image.png', cv2.IMREAD_UNCHANGED)
if surface_img.shape[2] == 4:  # Nếu có alpha channel, chuyển sang RGB
    surface_img = cv2.cvtColor(surface_img, cv2.COLOR_BGRA2RGB)
else:
    surface_img = cv2.cvtColor(surface_img, cv2.COLOR_BGR2RGB)

# Resize nếu cần cho YOLO
yolo_input_img = cv2.resize(surface_img, (416, 416))

# Hiển thị lại để kiểm tra
plt.imshow(yolo_input_img)
plt.title("2D Image Rendered from 3D Interpolated Surface (View-based)")
plt.axis('off')
plt.show()

# Lưu ảnh input nếu cần
cv2.imwrite('yolo_surface_input.png', cv2.cvtColor(yolo_input_img, cv2.COLOR_RGB2BGR))