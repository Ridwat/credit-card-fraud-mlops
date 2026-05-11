from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detection API")

model = joblib.load("models/xgboost_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = np.array(transaction.features).reshape(1, -1)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "result": "Fraud" if prediction == 1 else "Not Fraud"
    }