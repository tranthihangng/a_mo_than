from feature_extractor import CoalFeatureExtractor
from feature_groups import get_recommended_features
from feature_analysis import feature_selection_analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from feature_extractor import CoalFeatureExtractor
from feature_groups import get_recommended_features
from feature_analysis import feature_selection_analysis

def example_usage():
    """
    Ví dụ cách sử dụng để chuẩn bị dữ liệu cho ML
    """
    
    # Khởi tạo feature extractor
    extractor = CoalFeatureExtractor()
    
    # Giả sử bạn có nhiều ảnh và labels
    # images = [img1, img2, img3, ...]
    # masks = [mask1, mask2, mask3, ...]
    # labels = [0, 1, 2, 1, 0, ...]  # 0: than thấp, 1: than trung, 2: than cao
    
    # Trích xuất features cho tất cả ảnh
    # all_features = []
    # for img, mask in zip(images, masks):
    #     features = extractor.extract_ml_features(img, mask)
    #     all_features.append(features)
    
    # Tạo DataFrame
    # df = extractor.create_feature_dataframe(all_features, labels)
    
    # Feature selection
    # X = df.drop('label', axis=1)
    # y = df['label']
    # feature_analysis = feature_selection_analysis(X.values, y.values, X.columns.tolist())
    
    # Lấy top features
    # top_features = feature_analysis.head(20)['feature'].tolist()
    # X_selected = X[top_features]
    
    # Chuẩn hóa dữ liệu
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_selected)
    
    # Chia train/test
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    
    # Đánh giá
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    
    pass

if __name__ == "__main__":
    # In ra các nhóm đặc trưng được đề xuất
    feature_groups = get_recommended_features()
    
    print("DANH SÁCH ĐẶC TRƯNG CHO PHÂN LOẠI THAN")
    print("=" * 50)
    
    total_features = 0
    for group, features in feature_groups.items():
        print(f"\n{group} ({len(features)} features):")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
        total_features += len(features)
    
    print(f"\nTổng cộng: {total_features} đặc trưng được đề xuất")
    print("\nGhi chú:")
    print("- CRITICAL: Đặc trưng quan trọng nhất, nên có trong mọi model")
    print("- IMPORTANT: Đặc trưng quan trọng, cải thiện độ chính xác đáng kể")
    print("- USEFUL: Đặc trưng hữu ích, có thể cải thiện model")
    print("- SUPPLEMENTARY: Đặc trưng bổ sung, dùng khi cần model phức tạp hơn") 