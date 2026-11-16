
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
data = pd.read_csv("data/safety_data.csv")

X = data.drop("risk_level", axis=1)
y = data["risk_level"]

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 모델 학습
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 모델 평가
predictions = model.predict(X_test)

print("Model Training Complete!")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# 모델 저장
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/risk_model.pkl")

print("Model saved successfully!")
