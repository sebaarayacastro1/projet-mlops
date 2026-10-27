from pathlib import Path

import joblib
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from src.features.regroupement import MAX_SECURITY_INDEX, MAX_STREAMING_PACK

ROOT = Path(__file__).resolve().parent.parent
#preprocessor = joblib.load(ROOT / "models" / "preprocessor.joblib")
model = joblib.load(ROOT / "models" / "model.joblib")

app = FastAPI(title="Telco Churn API")
YesNo = Literal["Yes", "No"]

THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

if not 0.0 <= THRESHOLD <= 1.0:
    raise ValueError(
        "CHURN_THRESHOLD doit être compris entre 0 et 1."
    )


class Customer(BaseModel):
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    InternetService: Literal["DSL", "Fiber optic", "No"]
    Security_Index: int = Field(
        ge=0,
        le=MAX_SECURITY_INDEX,
        description=(
            "Nombre de services de sécurité souscrits parmi OnlineSecurity, "
            "OnlineBackup, DeviceProtection et TechSupport."
        ),
    )
    Streaming_Pack: int = Field(
        ge=0,
        le=MAX_STREAMING_PACK,
        description="Nombre de services de streaming souscrits parmi StreamingTV et StreamingMovies.",
    )
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def indices_coherents(self):
        if self.InternetService == "No" and (self.Security_Index or self.Streaming_Pack):
            raise ValueError(
                "Sans service Internet, Security_Index et Streaming_Pack doivent valoir 0."
            )
        return self


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    return {"churn": proba >= THRESHOLD, "churn_probability": round(proba, 4)}