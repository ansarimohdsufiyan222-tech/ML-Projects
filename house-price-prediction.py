import pandas as pd 
from sklearn.linear_model import LinearRegression

data = {
    "Area" : [600,800,1000,1200,1500,1800,2000,2200,2500],
    "Price" : [30,40,50,60,75,90,100,110,125]
}

df = pd.DataFrame(data)

X = df[["Area"]]
Y = df["Price"]

model = LinearRegression()
model.fit(X,Y)

price = model.predict([[1700]])

print("Predicted House Price:",price[0], "Lakhs")