import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def feature_selection_analysis(X, y, feature_names, top_k=20):
    """
    Phân tích và chọn đặc trưng quan trọng nhất
    """
    # Method 1: F-score
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X, y)
    f_scores = f_selector.scores_
    
    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Method 3: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Tạo DataFrame kết quả
    results = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'mutual_info': mi_scores,
        'rf_importance': rf_importance
    })
    
    # Normalize scores
    results['f_score_norm'] = (results['f_score'] - results['f_score'].min()) / (results['f_score'].max() - results['f_score'].min())
    results['mutual_info_norm'] = (results['mutual_info'] - results['mutual_info'].min()) / (results['mutual_info'].max() - results['mutual_info'].min())
    results['rf_importance_norm'] = results['rf_importance']
    
    # Tính điểm tổng hợp
    results['combined_score'] = (results['f_score_norm'] + results['mutual_info_norm'] + results['rf_importance_norm']) / 3
    
    # Sắp xếp theo điểm tổng hợp
    results = results.sort_values('combined_score', ascending=False)
    
    # Vẽ biểu đồ top features
    plt.figure(figsize=(12, 8))
    top_features = results.head(top_k)
    
    plt.subplot(2, 2, 1)
    plt.barh(range(len(top_features)), top_features['f_score_norm'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('F-Score')
    plt.gca().invert_yaxis()
    
    plt.subplot(2, 2, 2)
    plt.barh(range(len(top_features)), top_features['mutual_info_norm'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Mutual Information')
    plt.gca().invert_yaxis()
    
    plt.subplot(2, 2, 3)
    plt.barh(range(len(top_features)), top_features['rf_importance_norm'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Random Forest Importance')
    plt.gca().invert_yaxis()
    
    plt.subplot(2, 2, 4)
    plt.barh(range(len(top_features)), top_features['combined_score'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Combined Score')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return results 