import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("heart_disease.csv")

# Features and Target
X = df[[
    "Age",
    "BloodPressure",
    "Cholesterol",
    "MaxHeartRate",
    "BloodSugar"
]]

y = df["HeartDisease"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
print("Accuracy :", accuracy_score(y_test, prediction))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, prediction))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# Test New Patient
patient = pd.DataFrame({
    "Age": [55],
    "BloodPressure": [145],
    "Cholesterol": [260],
    "MaxHeartRate": [120],
    "BloodSugar": [1]
})

result = model.predict(patient)

if result[0] == 1:
    print("\nPrediction : Heart Disease Detected")
else:
    print("\nPrediction : No Heart Disease")