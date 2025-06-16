import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import disk, erosion, dilation, opening, closing
import seaborn as sns

def extract_roi_features(image_path, points):
    """
    Trích xuất đặc trưng từ vùng ROI của ảnh than
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return None

    # Tạo mask cho ROI
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Cắt ROI
    roi = cv2.bitwise_and(image, image, mask=mask)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Chỉ lấy các pixel bên trong ROI (khác 0)
    roi_pixels = gray_roi[mask == 255]

    features = {}

    # 1. ĐẶC TRƯNG THỐNG KÊ CƠ BẢN
    features['mean_intensity'] = np.mean(roi_pixels)
    features['std_intensity'] = np.std(roi_pixels)
    features['min_intensity'] = np.min(roi_pixels)
    features['max_intensity'] = np.max(roi_pixels)
    features['median_intensity'] = np.median(roi_pixels)
    features['range_intensity'] = features['max_intensity'] - features['min_intensity']
    features['coefficient_variation'] = features['std_intensity'] / features['mean_intensity']

    # 2. ĐẶC TRƯNG HISTOGRAM
    hist, bins = np.histogram(roi_pixels, bins=256, range=(0, 256))
    features['histogram_peak'] = np.argmax(hist)  # Giá trị độ sáng phổ biến nhất
    features['histogram_entropy'] = -np.sum(hist/np.sum(hist) * np.log2(hist/np.sum(hist) + 1e-10))

    # Phân tích phân bố độ sáng
    dark_pixels = np.sum(roi_pixels < 85)  # Pixel tối
    medium_pixels = np.sum((roi_pixels >= 85) & (roi_pixels < 170))  # Pixel trung bình
    bright_pixels = np.sum(roi_pixels >= 170)  # Pixel sáng
    total_pixels = len(roi_pixels)

    features['dark_ratio'] = dark_pixels / total_pixels
    features['medium_ratio'] = medium_pixels / total_pixels
    features['bright_ratio'] = bright_pixels / total_pixels

    # 3. ĐẶC TRƯNG TEXTURE (Gray-Level Co-occurrence Matrix)
    # Chuẩn hóa ảnh về 0-63 để tính GLCM
    roi_normalized = ((gray_roi / 255.0) * 63).astype(np.uint8)
    roi_normalized = roi_normalized * (mask // 255)  # Chỉ lấy vùng ROI

    # Tính GLCM với nhiều hướng và khoảng cách
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm_features = {}
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(roi_normalized, distances=[distance],
                             angles=[angle], levels=64, symmetric=True, normed=True)

            key_prefix = f'd{distance}_a{int(angle*180/np.pi)}'
            glcm_features[f'{key_prefix}_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            glcm_features[f'{key_prefix}_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            glcm_features[f'{key_prefix}_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            glcm_features[f'{key_prefix}_energy'] = graycoprops(glcm, 'energy')[0, 0]
            glcm_features[f'{key_prefix}_correlation'] = graycoprops(glcm, 'correlation')[0, 0]

    # Tính trung bình các đặc trưng GLCM
    features['avg_contrast'] = np.mean([v for k, v in glcm_features.items() if 'contrast' in k])
    features['avg_dissimilarity'] = np.mean([v for k, v in glcm_features.items() if 'dissimilarity' in k])
    features['avg_homogeneity'] = np.mean([v for k, v in glcm_features.items() if 'homogeneity' in k])
    features['avg_energy'] = np.mean([v for k, v in glcm_features.items() if 'energy' in k])
    features['avg_correlation'] = np.mean([v for k, v in glcm_features.items() if 'correlation' in k])

    # 4. ĐẶC TRƯNG HÌNH THÁI (Morphological Features)
    # Threshold để tạo ảnh nhị phân
    thresh = threshold_otsu(roi_pixels)
    binary_roi = (gray_roi > thresh).astype(np.uint8) * (mask // 255)

    # Đếm các thành phần liên thông
    labeled_image = label(binary_roi)
    regions = regionprops(labeled_image)

    if regions:
        areas = [region.area for region in regions]
        features['num_components'] = len(regions)
        features['avg_component_area'] = np.mean(areas)
        features['total_component_area'] = np.sum(areas)
        features['largest_component_area'] = np.max(areas)
        features['component_area_std'] = np.std(areas)

        # Tỷ lệ diện tích đối tượng so với ROI
        features['object_area_ratio'] = features['total_component_area'] / np.sum(mask // 255)
    else:
        features['num_components'] = 0
        features['avg_component_area'] = 0
        features['total_component_area'] = 0
        features['largest_component_area'] = 0
        features['component_area_std'] = 0
        features['object_area_ratio'] = 0

    # 5. ĐẶC TRƯNG GRADIENT VÀ EDGE
    # Tính gradient
    gradient = sobel(gray_roi * (mask // 255))
    features['avg_gradient'] = np.mean(gradient[mask == 255])
    features['gradient_std'] = np.std(gradient[mask == 255])

    # Đếm edge pixels
    edge_threshold = np.percentile(gradient[mask == 255], 90)
    edge_pixels = np.sum(gradient > edge_threshold)
    features['edge_density'] = edge_pixels / np.sum(mask // 255)

    # 6. ĐẶC TRƯNG PHÂN TÍCH KHÔNG GIAN
    # Tính độ biến thiên không gian
    y_coords, x_coords = np.where(mask == 255)
    intensities = gray_roi[y_coords, x_coords]

    # Phân tích theo hướng ngang và dọc
    features['horizontal_variation'] = np.std([np.mean(intensities[x_coords == x])
                                            for x in np.unique(x_coords) if len(intensities[x_coords == x]) > 1])
    features['vertical_variation'] = np.std([np.mean(intensities[y_coords == y])
                                           for y in np.unique(y_coords) if len(intensities[y_coords == y]) > 1])

    return features, roi, gray_roi, mask

# def analyze_coal_quality(features):
#     """
#     Phân tích chất lượng than dựa trên các đặc trưng
#     """
#     analysis = {}

#     # Phân tích độ đồng nhất
#     if features['coefficient_variation'] < 0.3:
#         analysis['uniformity'] = "Cao - Than đồng nhất"
#     elif features['coefficient_variation'] < 0.5:
#         analysis['uniformity'] = "Trung bình - Than khá đồng nhất"
#     else:
#         analysis['uniformity'] = "Thấp - Than không đồng nhất"

#     # Phân tích mật độ (dựa trên độ sáng trung bình)
#     if features['mean_intensity'] < 80:
#         analysis['density'] = "Cao - Than mật độ cao"
#     elif features['mean_intensity'] < 150:
#         analysis['density'] = "Trung bình - Than mật độ trung bình"
#     else:
#         analysis['density'] = "Thấp - Than mật độ thấp hoặc có tạp chất"

#     # Phân tích texture
#     if features['avg_homogeneity'] > 0.8:
#         analysis['texture'] = "Mịn - Bề mặt đồng nhất"
#     elif features['avg_homogeneity'] > 0.6:
#         analysis['texture'] = "Trung bình - Bề mặt khá đồng nhất"
#     else:
#         analysis['texture'] = "Thô - Bề mặt không đồng nhất"

#     # Phân tích tỷ lệ than tối/sáng
#     if features['dark_ratio'] > 0.7:
#         analysis['coal_content'] = "Cao - Chủ yếu là than"
#     elif features['dark_ratio'] > 0.4:
#         analysis['coal_content'] = "Trung bình - Than lẫn tạp chất"
#     else:
#         analysis['coal_content'] = "Thấp - Nhiều tạp chất hoặc không phải than"

#     return analysis

def visualize_analysis(image, roi, gray_roi, mask, features, analysis, points):
    """
    Trực quan hóa kết quả phân tích
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Ảnh gốc với ROI
    axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0,0].plot(np.append(points[:, 0], points[0, 0]),
                   np.append(points[:, 1], points[0, 1]), 'r-', linewidth=2)
    axes[0,0].set_title('Ảnh gốc với ROI')
    axes[0,0].axis('off')

    # ROI grayscale
    axes[0,1].imshow(gray_roi, cmap='gray')
    axes[0,1].set_title('ROI - Grayscale')
    axes[0,1].axis('off')

    # Histogram
    roi_pixels = gray_roi[mask == 255]
    axes[0,2].hist(roi_pixels, bins=50, alpha=0.7, color='blue')
    axes[0,2].axvline(features['mean_intensity'], color='red', linestyle='--',
                      label=f'Mean: {features["mean_intensity"]:.1f}')
    axes[0,2].axvline(features['histogram_peak'], color='green', linestyle='--',
                      label=f'Peak: {features["histogram_peak"]:.1f}')
    axes[0,2].set_title('Histogram độ sáng')
    axes[0,2].set_xlabel('Độ sáng')
    axes[0,2].set_ylabel('Tần suất')
    axes[0,2].legend()

    # 3D surface plot
    y, x = np.where(mask == 255)
    z = gray_roi[y, x]
    ax_3d = fig.add_subplot(2, 3, 4, projection='3d')
    scatter = ax_3d.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=1)
    ax_3d.set_title('Bề mặt 3D')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Độ sáng')
    plt.colorbar(scatter, ax=ax_3d, shrink=0.5)

    # Biểu đồ phân tích tỷ lệ
    categories = ['Tối\n(Than)', 'Trung bình', 'Sáng\n(Tạp chất)']
    ratios = [features['dark_ratio'], features['medium_ratio'], features['bright_ratio']]
    colors = ['black', 'gray', 'white']
    bars = axes[1,1].bar(categories, ratios, color=colors, edgecolor='black')
    axes[1,1].set_title('Phân bố độ sáng')
    axes[1,1].set_ylabel('Tỷ lệ')
    axes[1,1].set_ylim(0, 1)

    # Thêm giá trị lên các cột
    for bar, ratio in zip(bars, ratios):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{ratio:.2f}', ha='center', va='bottom')

    # Bảng thông tin đặc trưng
    axes[1,2].axis('off')
    info_text = f"""
    ĐẶC TRƯNG CHÍNH:
    • Độ sáng TB: {features['mean_intensity']:.1f}
    • Độ lệch chuẩn: {features['std_intensity']:.1f}
    • Hệ số biến thiên: {features['coefficient_variation']:.3f}
    • Entropy: {features['histogram_entropy']:.3f}
    • Độ đồng nhất: {features['avg_homogeneity']:.3f}
    • Độ tương phản: {features['avg_contrast']:.3f}
    • Tỷ lệ than: {features['dark_ratio']:.2%}

    ĐÁNH GIÁ:
    • Độ đồng nhất: {analysis['uniformity']}
    • Mật độ: {analysis['density']}
    • Texture: {analysis['texture']}
    • Hàm lượng than: {analysis['coal_content']}
    """
    axes[1,2].text(0.05, 0.95, info_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    return fig

# Hàm chính để chạy phân tích
def run_coal_analysis(image_path, points):
    """
    Chạy toàn bộ quá trình phân tích than
    """
    print("Đang trích xuất đặc trưng...")
    result = extract_roi_features(image_path, points)

    if result is None:
        return

    features, roi, gray_roi, mask = result

    print("Đang phân tích chất lượng...")
    analysis = analyze_coal_quality(features)

    print("Đang tạo biểu đồ...")
    image = cv2.imread(image_path)
    visualize_analysis(image, roi, gray_roi, mask, features, analysis, points)

    # In kết quả chi tiết
    print("\n" + "="*50)
    print("KẾT QUÁ PHÂN TÍCH THAN")
    print("="*50)

    print("\n1. ĐẶC TRƯNG THỐNG KÊ:")
    print(f"   - Độ sáng trung bình: {features['mean_intensity']:.2f}")
    print(f"   - Độ lệch chuẩn: {features['std_intensity']:.2f}")
    print(f"   - Hệ số biến thiên: {features['coefficient_variation']:.3f}")
    print(f"   - Khoảng giá trị: {features['range_intensity']:.0f}")

    print("\n2. PHÂN BỐ ĐỘ SÁNG:")
    print(f"   - Tỷ lệ pixel tối (than): {features['dark_ratio']:.2%}")
    print(f"   - Tỷ lệ pixel trung bình: {features['medium_ratio']:.2%}")
    print(f"   - Tỷ lệ pixel sáng (tạp chất): {features['bright_ratio']:.2%}")

    print("\n3. ĐẶC TRƯNG TEXTURE:")
    print(f"   - Độ đồng nhất: {features['avg_homogeneity']:.3f}")
    print(f"   - Độ tương phản: {features['avg_contrast']:.3f}")
    print(f"   - Độ tương quan: {features['avg_correlation']:.3f}")

    print("\n4. ĐÁNH GIÁ CHẤT LƯỢNG:")
    for key, value in analysis.items():
        print(f"   - {key.replace('_', ' ').title()}: {value}")

    return features, analysis

# Sử dụng:
image_path = '/content/hiv00007_000007.jpg'
points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]], np.int32)
features, analysis = run_coal_analysis(image_path, points)