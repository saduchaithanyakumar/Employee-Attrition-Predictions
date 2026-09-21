from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline.joblib"

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load(MODEL)


class Customer(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    contract: str
    tech_support: str
    internet_service: str
    payment_method: str


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(data)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "prediction": "Yes" if probability >= 0.5 else "No",
    }
