from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="welcome to this api")

model = joblib.load("home_price_prediction.pkl")

@app.get("/")
def home():
    return {"message": "welcome to this api"}

@app.post("/predict")
def predict(area: float, room: int):

    X_new = pd.DataFrame([[area, room]], columns=["area", "room"])

    prediction = model.predict(X_new)[0]

    return {"predicted price": round(prediction, 2)}