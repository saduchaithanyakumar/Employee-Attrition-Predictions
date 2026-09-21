# Customer Churn Prediction & Analytics Platform

An end-to-end machine learning project for analyzing customer behavior and predicting churn probability.

## Tech Stack
- Python
- Pandas / NumPy
- Scikit-Learn
- SQL
- FastAPI
- Docker
- Matplotlib / Seaborn

## Project Structure
```text
customer-churn-ml/
├── api/
│   └── app.py
├── data/
│   └── customer_churn.csv
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload
```

Open the API docs at `http://127.0.0.1:8000/docs`.

## Model
The training pipeline performs preprocessing, train/test splitting, model training and evaluation. The current baseline uses Logistic Regression with one-hot encoding and scaling.

## Important
The included CSV is a small synthetic starter dataset intended for development/demo purposes. Replace it with a larger public dataset and add your own analysis before presenting this project as completed work.
