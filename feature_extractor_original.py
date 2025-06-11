import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel, sobel_h, sobel_v, threshold_otsu
from skimage.measure import regionprops, label
from skimage.morphology import disk, erosion, dilation, opening, closing

class CoalFeatureExtractor:
    """
    Class để trích xuất đặc trưng cho phân loại than
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def extract_ml_features(self, image, roi_mask):
        """
        Trích xuất tất cả đặc trưng cần thiết cho ML
        Returns: dict chứa tất cả features
        """
        features = {}
        
        # Lấy pixels trong ROI
        roi_pixels = image[roi_mask == 255]
        
        # ==================== NHÓM 1: ĐẶC TRƯNG THỐNG KÊ CƠ BẢN ====================
        features.update(self._extract_basic_stats(roi_pixels))
        
        # ==================== NHÓM 2: ĐẶC TRƯNG PERCENTILE ====================
        features.update(self._extract_percentile_features(roi_pixels))
        
        # ==================== NHÓM 3: ĐẶC TRƯNG PHÂN BỐ THEO NGƯỠNG ====================
        features.update(self._extract_threshold_features(roi_pixels))
        
        # ==================== NHÓM 4: ĐẶC TRƯNG HISTOGRAM ====================
        features.update(self._extract_histogram_features(roi_pixels))
        
        # ==================== NHÓM 5: ĐẶC TRƯNG TEXTURE (GLCM) ====================
        features.update(self._calculate_glcm_features(image, roi_mask))
        
        # ==================== NHÓM 6: ĐẶC TRƯNG GRADIENT VÀ EDGE ====================
        features.update(self._calculate_gradient_features(image, roi_mask))
        
        # ==================== NHÓM 7: ĐẶC TRƯNG HÌNH THÁI ====================
        features.update(self._calculate_morphological_features(image, roi_mask))
        
        # ==================== NHÓM 8: ĐẶC TRƯNG KHÔNG GIAN ====================
        features.update(self._calculate_spatial_features(image, roi_mask))
        
        return features

    def _extract_basic_stats(self, roi_pixels):
        """Trích xuất các đặc trưng thống kê cơ bản"""
        features = {}
        features['mean_intensity'] = np.mean(roi_pixels)
        features['std_intensity'] = np.std(roi_pixels)
        features['var_intensity'] = np.var(roi_pixels)
        features['min_intensity'] = np.min(roi_pixels)
        features['max_intensity'] = np.max(roi_pixels)
        features['median_intensity'] = np.median(roi_pixels)
        features['range_intensity'] = features['max_intensity'] - features['min_intensity']
        features['iqr_intensity'] = np.percentile(roi_pixels, 75) - np.percentile(roi_pixels, 25)
        features['coefficient_variation'] = features['std_intensity'] / (features['mean_intensity'] + 1e-8)
        features['skewness'] = self._calculate_skewness(roi_pixels)
        features['kurtosis'] = self._calculate_kurtosis(roi_pixels)
        return features

    def _extract_percentile_features(self, roi_pixels):
        """Trích xuất các đặc trưng percentile"""
        features = {}
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            features[f'percentile_{p}'] = np.percentile(roi_pixels, p)
        return features

    def _extract_threshold_features(self, roi_pixels):
        """Trích xuất các đặc trưng dựa trên ngưỡng"""
        features = {}
        
        # Phân chia 3 mức cơ bản
        dark_threshold = 85
        bright_threshold = 170
        
        features['dark_ratio'] = np.sum(roi_pixels < dark_threshold) / len(roi_pixels)
        features['medium_ratio'] = np.sum((roi_pixels >= dark_threshold) & 
                                        (roi_pixels < bright_threshold)) / len(roi_pixels)
        features['bright_ratio'] = np.sum(roi_pixels >= bright_threshold) / len(roi_pixels)
        
        # Phân chia 5 mức chi tiết hơn
        thresholds = [51, 102, 153, 204]  # Chia 0-255 thành 5 phần
        for i, thresh in enumerate(thresholds):
            if i == 0:
                features[f'level_{i}_ratio'] = np.sum(roi_pixels < thresh) / len(roi_pixels)
            else:
                features[f'level_{i}_ratio'] = np.sum((roi_pixels >= thresholds[i-1]) & 
                                                    (roi_pixels < thresh)) / len(roi_pixels)
        features[f'level_4_ratio'] = np.sum(roi_pixels >= thresholds[-1]) / len(roi_pixels)
        
        # Phân chia theo Otsu threshold
        otsu_thresh = self._otsu_threshold(roi_pixels)
        features['otsu_threshold'] = otsu_thresh
        features['below_otsu_ratio'] = np.sum(roi_pixels < otsu_thresh) / len(roi_pixels)
        features['above_otsu_ratio'] = np.sum(roi_pixels >= otsu_thresh) / len(roi_pixels)
        
        return features

    def _extract_histogram_features(self, roi_pixels):
        """Trích xuất các đặc trưng histogram"""
        features = {}
        hist, bins = np.histogram(roi_pixels, bins=32, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        
        features['hist_peak'] = np.argmax(hist) * 8  # Vì chia 32 bins
        features['hist_entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        features['hist_uniformity'] = np.sum(hist_normalized ** 2)
        features['hist_contrast'] = np.sum([(i - j) ** 2 * hist_normalized[i] * hist_normalized[j] 
                                          for i in range(len(hist_normalized)) 
                                          for j in range(len(hist_normalized))])
        
        # Moments của histogram
        features['hist_mean'] = np.sum([i * hist_normalized[i] for i in range(len(hist_normalized))])
        features['hist_variance'] = np.sum([(i - features['hist_mean']) ** 2 * hist_normalized[i] 
                                          for i in range(len(hist_normalized))])
        features['hist_skewness'] = np.sum([(i - features['hist_mean']) ** 3 * hist_normalized[i] 
                                          for i in range(len(hist_normalized))]) / (features['hist_variance'] ** 1.5)
        features['hist_kurtosis'] = np.sum([(i - features['hist_mean']) ** 4 * hist_normalized[i] 
                                          for i in range(len(hist_normalized))]) / (features['hist_variance'] ** 2)
        
        return features

    def _calculate_skewness(self, data):
        """Tính độ xiên"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data):
        """Tính độ nhọn"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _otsu_threshold(self, data):
        """Tính Otsu threshold"""
        hist, bins = np.histogram(data, bins=256, range=(0, 256))
        hist = hist.astype(float)
        
        # Normalize histogram
        hist /= hist.sum()
        
        # Compute cumulative sums
        w0 = np.cumsum(hist)
        w1 = 1 - w0
        
        # Compute cumulative means
        mu0 = np.cumsum(hist * np.arange(256)) / (w0 + 1e-10)
        mu1 = (np.sum(hist * np.arange(256)) - np.cumsum(hist * np.arange(256))) / (w1 + 1e-10)
        
        # Compute between-class variance
        variance_between = w0 * w1 * (mu0 - mu1) ** 2
        
        # Find optimal threshold
        threshold = np.argmax(variance_between)
        return threshold
    
    def _calculate_glcm_features(self, image, roi_mask):
        """Tính các đặc trưng GLCM"""
        # Chuẩn hóa ảnh về 0-31 (32 levels)
        roi_normalized = ((image / 255.0) * 31).astype(np.uint8)
        roi_normalized = roi_normalized * (roi_mask // 255)
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = {}
        
        all_values = {prop: [] for prop in glcm_props}
        
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(roi_normalized, distances=[distance], 
                                 angles=[angle], levels=32, symmetric=True, normed=True)
                
                for prop in glcm_props:
                    if prop == 'ASM':
                        value = graycoprops(glcm, 'ASM')[0, 0]
                    else:
                        value = graycoprops(glcm, prop)[0, 0]
                    all_values[prop].append(value)
        
        # Tính thống kê cho mỗi property
        for prop in glcm_props:
            values = all_values[prop]
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
            features[f'glcm_{prop}_min'] = np.min(values)
            features[f'glcm_{prop}_max'] = np.max(values)
        
        return features
    
    def _calculate_gradient_features(self, image, roi_mask):
        """Tính các đặc trưng gradient"""
        features = {}
        
        # Sobel gradient
        gradient_sobel = sobel(image * (roi_mask // 255))
        roi_gradient = gradient_sobel[roi_mask == 255]
        
        features['gradient_mean'] = np.mean(roi_gradient)
        features['gradient_std'] = np.std(roi_gradient)
        features['gradient_max'] = np.max(roi_gradient)
        features['gradient_min'] = np.min(roi_gradient)
        
        # Directional gradients
        grad_h = sobel_h(image * (roi_mask // 255))
        grad_v = sobel_v(image * (roi_mask // 255))
        
        features['gradient_h_mean'] = np.mean(grad_h[roi_mask == 255])
        features['gradient_v_mean'] = np.mean(grad_v[roi_mask == 255])
        features['gradient_h_std'] = np.std(grad_h[roi_mask == 255])
        features['gradient_v_std'] = np.std(grad_v[roi_mask == 255])
        
        # Edge density
        edge_threshold = np.percentile(roi_gradient, 90)
        features['edge_density'] = np.sum(roi_gradient > edge_threshold) / len(roi_gradient)
        
        # Gradient magnitude và direction
        gradient_magnitude = np.sqrt(grad_h**2 + grad_v**2)
        features['gradient_magnitude_mean'] = np.mean(gradient_magnitude[roi_mask == 255])
        features['gradient_magnitude_std'] = np.std(gradient_magnitude[roi_mask == 255])
        
        return features
    
    def _calculate_morphological_features(self, image, roi_mask):
        """Tính các đặc trưng hình thái"""
        features = {}
        
        # Tạo ảnh nhị phân
        roi_pixels = image[roi_mask == 255]
        if len(roi_pixels) > 0:
            thresh = threshold_otsu(roi_pixels)
            binary_roi = (image > thresh).astype(np.uint8) * (roi_mask // 255)
            
            # Đếm các thành phần liên thông
            labeled_image = label(binary_roi)
            regions = regionprops(labeled_image)
            
            if regions:
                areas = [region.area for region in regions]
                eccentricities = [region.eccentricity for region in regions]
                solidities = [region.solidity for region in regions]
                
                features['num_components'] = len(regions)
                features['total_area'] = np.sum(areas)
                features['avg_area'] = np.mean(areas)
                features['area_std'] = np.std(areas)
                features['max_area'] = np.max(areas)
                features['min_area'] = np.min(areas)
                
                features['avg_eccentricity'] = np.mean(eccentricities)
                features['avg_solidity'] = np.mean(solidities)
                
                # Tỷ lệ diện tích
                roi_area = np.sum(roi_mask // 255)
                features['object_area_ratio'] = features['total_area'] / roi_area
                
                # Morphological operations
                selem = disk(3)
                eroded = erosion(binary_roi, selem)
                dilated = dilation(binary_roi, selem)
                opened = opening(binary_roi, selem)
                closed = closing(binary_roi, selem)
                
                features['erosion_ratio'] = np.sum(eroded) / np.sum(binary_roi) if np.sum(binary_roi) > 0 else 0
                features['dilation_ratio'] = np.sum(dilated) / np.sum(binary_roi) if np.sum(binary_roi) > 0 else 0
                features['opening_ratio'] = np.sum(opened) / np.sum(binary_roi) if np.sum(binary_roi) > 0 else 0
                features['closing_ratio'] = np.sum(closed) / np.sum(binary_roi) if np.sum(binary_roi) > 0 else 0
            else:
                # Nếu không có thành phần nào
                for key in ['num_components', 'total_area', 'avg_area', 'area_std', 
                           'max_area', 'min_area', 'avg_eccentricity', 'avg_solidity',
                           'object_area_ratio', 'erosion_ratio', 'dilation_ratio', 
                           'opening_ratio', 'closing_ratio']:
                    features[key] = 0
        
        return features
    
    def _calculate_spatial_features(self, image, roi_mask):
        """Tính các đặc trưng không gian"""
        features = {}
        
        # Lấy tọa độ và giá trị intensity
        y_coords, x_coords = np.where(roi_mask == 255)
        intensities = image[y_coords, x_coords]
        
        if len(intensities) > 0:
            # Phân tích theo hướng ngang
            unique_x = np.unique(x_coords)
            if len(unique_x) > 1:
                x_means = [np.mean(intensities[x_coords == x]) for x in unique_x 
                          if len(intensities[x_coords == x]) > 0]
                features['horizontal_variation'] = np.std(x_means) if len(x_means) > 1 else 0
                features['horizontal_range'] = np.max(x_means) - np.min(x_means) if len(x_means) > 1 else 0
            else:
                features['horizontal_variation'] = 0
                features['horizontal_range'] = 0
            
            # Phân tích theo hướng dọc
            unique_y = np.unique(y_coords)
            if len(unique_y) > 1:
                y_means = [np.mean(intensities[y_coords == y]) for y in unique_y 
                          if len(intensities[y_coords == y]) > 0]
                features['vertical_variation'] = np.std(y_means) if len(y_means) > 1 else 0
                features['vertical_range'] = np.max(y_means) - np.min(y_means) if len(y_means) > 1 else 0
            else:
                features['vertical_variation'] = 0
                features['vertical_range'] = 0
            
            # Center of mass
            features['center_of_mass_x'] = np.average(x_coords, weights=intensities)
            features['center_of_mass_y'] = np.average(y_coords, weights=intensities)
            
            # Spatial moments
            features['spatial_moment_x'] = np.sum((x_coords - features['center_of_mass_x'])**2 * intensities)
            features['spatial_moment_y'] = np.sum((y_coords - features['center_of_mass_y'])**2 * intensities)
        else:
            for key in ['horizontal_variation', 'horizontal_range', 'vertical_variation', 
                       'vertical_range', 'center_of_mass_x', 'center_of_mass_y',
                       'spatial_moment_x', 'spatial_moment_y']:
                features[key] = 0
        
        return features
    
    def get_feature_vector(self, features_dict):
        """
        Chuyển đổi dictionary features thành vector cho ML
        """
        # Định nghĩa thứ tự features
        if not self.feature_names:
            self.feature_names = sorted(features_dict.keys())
        
        # Tạo vector theo thứ tự đã định
        vector = [features_dict.get(name, 0) for name in self.feature_names]
        return np.array(vector)
    
    def create_feature_dataframe(self, features_list, labels=None):
        """
        Tạo DataFrame từ danh sách features
        """
        if not features_list:
            return None
        
        # Đảm bảo tất cả features có cùng keys
        all_keys = set()
        for features in features_list:
            all_keys.update(features.keys())
        
        self.feature_names = sorted(all_keys)
        
        # Tạo ma trận features
        feature_matrix = []
        for features in features_list:
            vector = [features.get(name, 0) for name in self.feature_names]
            feature_matrix.append(vector)
        
        df = pd.DataFrame(feature_matrix, columns=self.feature_names)
        
        if labels is not None:
            df['label'] = labels
        
        return df 