from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline.joblib"

model = joblib.load(MODEL)

sample = pd.DataFrame([{
    "tenure": 6,
    "monthly_charges": 82.0,
    "total_charges": 492.0,
    "contract": "Month-to-month",
    "tech_support": "No",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
}])

probability = model.predict_proba(sample)[0, 1]
print(f"Churn probability: {probability:.2%}")
