import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")

# Features and Target
X = df[[
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income"
]]

y = df["loan_status"]

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

# Test New Customer
customer = pd.DataFrame({
    "person_income": [50000],
    "loan_amnt": [10000],
    "loan_int_rate": [12.5],
    "loan_percent_income": [0.20]
})

result = model.predict(customer)


if result[0] == 1:
    print("Prediction : High Credit Risk")
else:
    print("Prediction : Low Credit Risk")