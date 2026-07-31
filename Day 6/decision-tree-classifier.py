# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
print(data.head())

# -------------------------------
# Convert Categorical Data
# -------------------------------
le = LabelEncoder()

data["Employment_Status"] = le.fit_transform(data["Employment_Status"])
data["Existing_Loan"] = le.fit_transform(data["Existing_Loan"])
data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])

# Select Features and Target
# -------------------------------
X = data[["Age",
          "Monthly_Income",
          "Credit_Score",
          "Employment_Status",
          "Existing_Loan"]]

y = data["Loan_Approved"]

# Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


new_applicant = [[32, 65000, 740, 1, 0]]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Rejected")

# -------------------------------
# Display Decision Tree
# -------------------------------
plt.figure(figsize=(15,8))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.show()
