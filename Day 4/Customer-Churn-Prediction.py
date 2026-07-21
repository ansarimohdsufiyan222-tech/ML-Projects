import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("customer_churn_dataset.csv")

encoders = {}
for col in ["Gender","Partner","Contract","PaymentMethod","Churn"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, prediction))

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# Test New Customer
new_customer = pd.DataFrame({
    "Gender": [encoders["Gender"].transform(["Male"])[0]],
    "SeniorCitizen": [0],
    "Partner": [encoders["Partner"].transform(["Yes"])[0]],
    "Tenure": [24],
    "MonthlyCharges": [55.0],
    "TotalCharges": [1320.0],
    "Contract": [encoders["Contract"].transform(["One year"])[0]],
    "PaymentMethod": [encoders["PaymentMethod"].transform(["Credit card"])[0]]
})

result = model.predict(new_customer)

print("\nPrediction :", "Customer Will Churn" if result[0] == 1 else "Customer Will Stay")