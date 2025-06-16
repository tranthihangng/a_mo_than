import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Đọc dữ liệu
    df = pd.read_csv('merged_data_finalxyyz.csv')
    
    # Xác định X và y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode nhãn nếu là dạng text
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, 'label_encoder.pkl')

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Dự đoán trên tập train và test
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Đánh giá hiệu suất trên tập train
    print("=== Train Evaluation ===")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    print(pd.DataFrame(train_report).transpose())

    # Đánh giá hiệu suất trên tập test
    print("\n=== Test Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    print(pd.DataFrame(test_report).transpose())

    # Vẽ confusion matrix cho tập test
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_ if 'le' in locals() else None,
                yticklabels=le.classes_ if 'le' in locals() else None)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Data')
    plt.tight_layout()
    plt.show()

    # Lưu model
    joblib.dump(clf, 'coal_model.pkl')
    print('\nĐã lưu model vào file coal_model.pkl')

if __name__ == '__main__':
    main()
