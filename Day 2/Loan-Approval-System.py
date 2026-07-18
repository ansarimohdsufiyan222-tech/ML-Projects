import pandas as pd
from  sklearn.linear_model import LogisticRegression
#Data Set
data = {
    "Income": [20, 30, 40, 50, 60, 70, 80, 90, 100],
    "Age": [5, 10, 15, 20, 25, 30, 35, 40, 45],
    "Loan Approved": [0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Income", "Age"]]
Y = df["Loan Approved"]

model = LogisticRegression()
model.fit(X, Y)

# Customer Details

income = 20
age = 21


result = model.predict([[income, age]])

print("Customer Income:",income,"Lakhs")
print("Customer Age:",age,"Years")

if result[0] == 1:
    print("Loan Status: Loan Approved")
else:
    print("Loan Status: Loan Not Approved")