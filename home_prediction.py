import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

dataset = {
    "area": [50, 70, 80, 90, 100, 120],
    "room": [1, 2, 2, 2, 3, 3],
    "price": [3000, 4000, 4900, 5500, 6000, 6500]

}

df = pd.DataFrame(dataset)

X = df[["area", "room"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "home_price_prediction.pkl")

print("model saved as home_price_prediction.pkl")
