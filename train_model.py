import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # Đọc dữ liệu
    df = pd.read_csv('merged_data_finalxyyz.csv')
    
    # Xác định X và y
    X = df.iloc[:, :-1]  # Tất cả các cột trừ cột cuối
    y = df.iloc[:, -1]   # Cột cuối cùng là class

    # Encode nhãn nếu là dạng text
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, 'label_encoder.pkl')  # Lưu encoder nếu cần dùng lại

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')  # Lưu scaler để dùng khi predict

    # Chia train/test (tùy chọn, để kiểm tra model)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Khởi tạo và huấn luyện model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Đánh giá nhanh
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Lưu model
    joblib.dump(clf, 'coal_model.pkl')
    print('Đã lưu model vào file coal_model.pkl')

if __name__ == '__main__':
    main()