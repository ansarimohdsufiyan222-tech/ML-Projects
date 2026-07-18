import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dataset
data = {
    "Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 45, 50, 60, 72, 80, 88, 95, 100]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
Y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[7.5]])

print("Predicted Marks:", prediction[0])

test_prediction = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, test_prediction))