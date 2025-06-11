def get_recommended_features():
    """
    Trả về danh sách các đặc trưng được đề xuất theo mức độ quan trọng
    """
    
    # Nhóm đặc trưng theo mức độ quan trọng
    feature_groups = {
        'CRITICAL': [
            'mean_intensity',           # Độ sáng trung bình - quan trọng nhất
            'dark_ratio',              # Tỷ lệ than (pixel tối)
            'std_intensity',           # Độ đồng nhất
            'coefficient_variation',   # Độ biến thiên tương đối
            'median_intensity',        # Robust central tendency
            'percentile_25',          # Quartile thấp
            'percentile_75',          # Quartile cao
        ],
        
        'IMPORTANT': [
            'bright_ratio',           # Tỷ lệ tạp chất
            'hist_entropy',           # Độ phức tạp phân bố
            'glcm_contrast_mean',     # Texture contrast
            'glcm_homogeneity_mean',  # Texture homogeneity
            'gradient_mean',          # Độ thay đổi cục bộ
            'edge_density',           # Mật độ biên
            'otsu_threshold',         # Ngưỡng phân chia tự động
            'below_otsu_ratio',       # Tỷ lệ dưới ngưỡng Otsu
        ],
        
        'USEFUL': [
            'range_intensity',        # Khoảng giá trị
            'iqr_intensity',         # Interquartile range
            'skewness',              # Độ xiên phân bố
            'hist_peak',             # Đỉnh histogram
            'glcm_energy_mean',      # Texture energy
            'glcm_correlation_mean', # Texture correlation
            'gradient_std',          # Biến thiên gradient
            'num_components',        # Số thành phần
            'object_area_ratio',     # Tỷ lệ diện tích đối tượng
        ],
        
        'SUPPLEMENTARY': [
            'level_0_ratio',         # Mức 0 (rất tối)
            'level_1_ratio',         # Mức 1
            'level_2_ratio',         # Mức 2
            'level_3_ratio',         # Mức 3
            'level_4_ratio',         # Mức 4 (rất sáng)
            'horizontal_variation',   # Biến thiên ngang
            'vertical_variation',     # Biến thiên dọc
            'gradient_magnitude_mean', # Độ lớn gradient
        ]
    }
    
    return feature_groups 