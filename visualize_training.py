import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style cho các biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_data_distribution(df, y):
    """Visualize phân phối dữ liệu"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution
    class_counts = Counter(y)
    axes[0, 0].bar(class_counts.keys(), class_counts.values(), alpha=0.7)
    axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    
    # Feature correlation heatmap (top 10 features)
    corr_matrix = df.iloc[:, :-1].corr()
    top_features = corr_matrix.abs().sum().nlargest(10).index
    sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', center=0,
                ax=axes[0, 1], fmt='.2f')
    axes[0, 1].set_title('Top 10 Features Correlation', fontsize=14, fontweight='bold')
    
    # Feature importance preview (using a quick RF fit)
    X_temp = StandardScaler().fit_transform(df.iloc[:, :-1])
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_temp.fit(X_temp, y)
    
    feature_importance = pd.DataFrame({
        'feature': df.columns[:-1],
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[1, 0].set_yticks(range(len(feature_importance)))
    axes[1, 0].set_yticklabels(feature_importance['feature'])
    axes[1, 0].set_title('Top 10 Feature Importance (Preview)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Importance')
    
    # Data statistics
    axes[1, 1].text(0.1, 0.8, f'Dataset Shape: {df.shape}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Number of Features: {df.shape[1]-1}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Number of Classes: {len(class_counts)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Missing Values: {df.isnull().sum().sum()}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Dataset Statistics', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_process(X_train, y_train, X_test, y_test):
    """Visualize quá trình học của Random Forest"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[0, 0].plot(train_sizes, train_mean, 'o-', label='Training Accuracy', linewidth=2)
    axes[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    axes[0, 0].plot(train_sizes, val_mean, 'o-', label='Validation Accuracy', linewidth=2)
    axes[0, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.3)
    axes[0, 0].set_xlabel('Training Set Size')
    axes[0, 0].set_ylabel('Accuracy Score')
    axes[0, 0].set_title('Learning Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation Curve (n_estimators)
    param_range = [10, 25, 50, 75, 100, 150, 200, 300]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42),
        X_train, y_train, param_name='n_estimators',
        param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[0, 1].plot(param_range, train_mean, 'o-', label='Training Accuracy', linewidth=2)
    axes[0, 1].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.3)
    axes[0, 1].plot(param_range, val_mean, 'o-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.3)
    axes[0, 1].set_xlabel('Number of Estimators')
    axes[0, 1].set_ylabel('Accuracy Score')
    axes[0, 1].set_title('Validation Curve (n_estimators)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Out-of-Bag Error Evolution
    oob_errors = []
    estimators_range = range(10, 201, 10)
    
    for n_est in estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42)
        rf.fit(X_train, y_train)
        oob_errors.append(1 - rf.oob_score_)
    
    axes[1, 0].plot(estimators_range, oob_errors, 'r-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Estimators')
    axes[1, 0].set_ylabel('Out-of-Bag Error Rate')
    axes[1, 0].set_title('OOB Error Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Progress Simulation
    rf_progressive = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)
    train_accuracies = []
    test_accuracies = []
    
    for i in range(1, 101):
        rf_progressive.n_estimators = i
        rf_progressive.fit(X_train, y_train)
        
        train_pred = rf_progressive.predict(X_train)
        test_pred = rf_progressive.predict(X_test)
        
        train_accuracies.append(accuracy_score(y_train, train_pred))
        test_accuracies.append(accuracy_score(y_test, test_pred))
    
    axes[1, 1].plot(range(1, 101), train_accuracies, label='Training Accuracy', linewidth=2)
    axes[1, 1].plot(range(1, 101), test_accuracies, label='Test Accuracy', linewidth=2)
    axes[1, 1].set_xlabel('Number of Trees Added')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Progressive Training Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_accuracies, test_accuracies

def plot_model_performance(clf, X_test, y_test, feature_names):
    """Visualize hiệu suất model"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[0, 1].set_yticks(range(len(feature_importance)))
    axes[0, 1].set_yticklabels(feature_importance['feature'])
    axes[0, 1].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Prediction Confidence Distribution
    y_proba = clf.predict_proba(X_test)
    max_proba = np.max(y_proba, axis=1)
    
    axes[1, 0].hist(max_proba, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(np.mean(max_proba), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(max_proba):.3f}')
    axes[1, 0].legend()
    
    # 4. Tree Depth Distribution
    tree_depths = [tree.tree_.max_depth for tree in clf.estimators_]
    
    axes[1, 1].hist(tree_depths, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Tree Depth')
    axes[1, 1].set_ylabel('Number of Trees')
    axes[1, 1].set_title('Tree Depth Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(np.mean(tree_depths), color='red', linestyle='--',
                      label=f'Mean Depth: {np.mean(tree_depths):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_trees(clf, feature_names, max_trees=3):
    """Visualize một số cây quyết định riêng lẻ"""
    fig, axes = plt.subplots(1, max_trees, figsize=(20, 8))
    
    for i in range(max_trees):
        plot_tree(clf.estimators_[i], 
                 feature_names=feature_names[:10],  # Chỉ hiển thị top 10 features
                 filled=True, 
                 rounded=True,
                 max_depth=3,  # Giới hạn độ sâu để dễ nhìn
                 ax=axes[i])
        axes[i].set_title(f'Decision Tree {i+1}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('individual_trees.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Đọc dữ liệu
    df = pd.read_csv('merged_data_finalxyyz.csv')
    
    # Xác định X và y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = X.columns.tolist()

    # Encode nhãn nếu là dạng text
    le = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, 'label_encoder.pkl')

    print("1. Visualizing data distribution...")
    plot_data_distribution(df, y)

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Visualizing learning process...")
    train_acc, test_acc = plot_learning_process(X_train, y_train, X_test, y_test)

    # Huấn luyện model chính
    print("3. Training final model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    clf.fit(X_train, y_train)

    # Đánh giá
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Final Accuracy: {accuracy:.4f}')
    print(f'OOB Score: {clf.oob_score_:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    print("4. Visualizing model performance...")
    plot_model_performance(clf, X_test, y_test, feature_names)

    print("5. Visualizing individual trees...")
    plot_individual_trees(clf, feature_names)

    # Lưu model
    joblib.dump(clf, 'coal_model.pkl')
    print('\nModel saved successfully!')
    
    # Tạo summary report
    print("\n" + "="*50)
    print("RANDOM FOREST TRAINING SUMMARY")
    print("="*50)
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {clf.oob_score_:.4f}")
    print(f"Average Tree Depth: {np.mean([tree.tree_.max_depth for tree in clf.estimators_]):.1f}")
    print(f"Total Parameters: ~{sum([tree.tree_.node_count for tree in clf.estimators_]):,}")

if __name__ == '__main__':
    main()