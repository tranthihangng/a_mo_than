import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import CoalFeatureExtractor
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FeatureVisualizer:
    def __init__(self):
        self.feature_extractor = CoalFeatureExtractor()
        
    def preprocess_image(self, image):
        """
        Chuyển đổi ảnh sang grayscale nếu cần
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
        
    def visualize_roi_features(self, image, roi_mask):
        """
        Trực quan hóa các đặc trưng của ROI
        """
        # Chuyển ảnh sang grayscale
        gray_image = self.preprocess_image(image)
        
        # Trích xuất features
        features = self.feature_extractor.extract_ml_features(gray_image, roi_mask)
        
        # Tạo figure với nhiều subplot
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Histogram ROI", "GLCM Features",
                "Gradient Features", "Morphological Features",
                "Spatial Features", "Statistical Features"
            )
        )
        
        # 1. Histogram
        roi_pixels = gray_image[roi_mask == 255]
        hist, bins = np.histogram(roi_pixels, bins=32, range=(0, 256))
        fig.add_trace(
            go.Bar(x=bins[:-1], y=hist, name="Histogram"),
            row=1, col=1
        )
        
        # 2. GLCM Features
        glcm_features = {k: v for k, v in features.items() if k.startswith('glcm_')}
        fig.add_trace(
            go.Bar(x=list(glcm_features.keys()), y=list(glcm_features.values()), name="GLCM"),
            row=1, col=2
        )
        
        # 3. Gradient Features
        gradient_features = {k: v for k, v in features.items() if k.startswith('gradient_')}
        fig.add_trace(
            go.Bar(x=list(gradient_features.keys()), y=list(gradient_features.values()), name="Gradient"),
            row=2, col=1
        )
        
        # 4. Morphological Features
        morph_features = {k: v for k, v in features.items() if k.startswith(('num_', 'total_', 'avg_', 'area_'))}
        fig.add_trace(
            go.Bar(x=list(morph_features.keys()), y=list(morph_features.values()), name="Morphological"),
            row=2, col=2
        )
        
        # 5. Spatial Features
        spatial_features = {k: v for k, v in features.items() if k.startswith(('horizontal_', 'vertical_', 'center_', 'spatial_'))}
        fig.add_trace(
            go.Bar(x=list(spatial_features.keys()), y=list(spatial_features.values()), name="Spatial"),
            row=3, col=1
        )
        
        # 6. Statistical Features
        stat_features = {k: v for k, v in features.items() if k in ['mean_intensity', 'std_intensity', 'var_intensity', 
                                                                   'min_intensity', 'max_intensity', 'median_intensity']}
        fig.add_trace(
            go.Bar(x=list(stat_features.keys()), y=list(stat_features.values()), name="Statistical"),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1200, width=1600, title_text="Feature Visualization")
        fig.show()
        
    def visualize_feature_correlation(self, features_list):
        """
        Trực quan hóa ma trận tương quan giữa các features
        """
        # Tạo DataFrame từ features
        df = self.feature_extractor.create_feature_dataframe(features_list)
        
        # Tính ma trận tương quan
        corr_matrix = df.corr()
        
        # Vẽ heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()
        
    def visualize_feature_distribution(self, features_list, labels=None):
        """
        Trực quan hóa phân phối của các features
        """
        # Tạo DataFrame từ features
        df = self.feature_extractor.create_feature_dataframe(features_list, labels)
        
        # Chuẩn hóa features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df.drop('label', axis=1))
        df_scaled = pd.DataFrame(features_scaled, columns=df.columns[:-1])
        if labels is not None:
            df_scaled['label'] = labels
            
        # Vẽ boxplot cho mỗi feature
        plt.figure(figsize=(20, 10))
        df_scaled.boxplot()
        plt.xticks(rotation=90)
        plt.title("Feature Distribution")
        plt.tight_layout()
        plt.show()
        
    def visualize_feature_importance(self, features_list, labels):
        """
        Trực quan hóa tầm quan trọng của các features
        """
        # Tạo DataFrame từ features
        df = self.feature_extractor.create_feature_dataframe(features_list, labels)
        
        # Tính tầm quan trọng của features dựa trên phương sai
        feature_importance = df.drop('label', axis=1).var()
        
        # Vẽ bar plot
        plt.figure(figsize=(15, 8))
        feature_importance.sort_values(ascending=False).plot(kind='bar')
        plt.title("Feature Importance (Variance)")
        plt.xlabel("Features")
        plt.ylabel("Variance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
    def visualize_roi_analysis(self, image, roi_mask):
        """
        Trực quan hóa phân tích ROI với các đặc trưng chính
        """
        # Chuyển ảnh sang grayscale
        gray_image = self.preprocess_image(image)
        
        # Tạo figure với subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Original ROI
        axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("Original Image with ROI")
        
        # 2. ROI Mask
        axes[0,1].imshow(roi_mask, cmap='gray')
        axes[0,1].set_title("ROI Mask")
        
        # 3. ROI Histogram
        roi_pixels = gray_image[roi_mask == 255]
        axes[1,0].hist(roi_pixels.ravel(), bins=32, range=(0, 256))
        axes[1,0].set_title("ROI Histogram")
        
        # 4. Feature Statistics
        features = self.feature_extractor.extract_ml_features(gray_image, roi_mask)
        stats = pd.Series({
            'Mean Intensity': features['mean_intensity'],
            'Std Intensity': features['std_intensity'],
            'Dark Ratio': features['dark_ratio'],
            'Bright Ratio': features['bright_ratio'],
            'Homogeneity': features['glcm_homogeneity_mean'],
            'Contrast': features['glcm_contrast_mean']
        })
        
        stats.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title("Key Features")
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    visualizer = FeatureVisualizer()
    
    # Load image and create ROI mask
    image = cv2.imread("hiv00007_000302.jpg")
    if image is None:
        print("Error: Could not load image")
        return
        
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    #roi_points = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], np.int32)
    # Tọa độ các điểm xác định vùng cần lấy (theo thứ tự: [x, y])
    roi_points = np.array([ [663, 839], [1201, 661], [1717, 663], [1785, 1021] ], np.int32)
    cv2.fillPoly(roi_mask, [roi_points], 255)
    
    # Visualize features
    visualizer.visualize_roi_features(image, roi_mask)
    visualizer.visualize_roi_analysis(image, roi_mask)

if __name__ == "__main__":
    main() 