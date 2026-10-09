from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
#preprocessor = joblib.load(ROOT / "models" / "preprocessor.joblib")
model = joblib.load(ROOT / "models" / "model.joblib")

app = FastAPI(title="Telco Churn API")
YesNo = Literal["Yes", "No"]
InternetExtra = Literal["Yes", "No", "No internet service"]

THRESHOLD = 0.5


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetExtra
    OnlineBackup: InternetExtra
    DeviceProtection: InternetExtra
    TechSupport: InternetExtra
    StreamingTV: InternetExtra
    StreamingMovies: InternetExtra
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    return {"churn": proba >= THRESHOLD, "churn_probability": round(proba, 4)}