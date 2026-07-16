import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["StudyHours", "Attendance"]]
Y = df["Pass"]

model = LogisticRegression()

model.fit(X, Y)

study_hours = 6
attendance = 6
result = model.predict([[study_hours, attendance]])

print("Study Hours:", study_hours)
print("Attendance:", attendance)

if result[0] == 1:
    print("Result : Pass")
else:
    print("Result : Fail")