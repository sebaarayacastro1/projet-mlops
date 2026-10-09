from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn"], bool)


def test_predict_invalid_category():
    r = client.post("/predict", json={**VALID, "Contract": "Weekly"})
    assert r.status_code == 422


def test_predict_missing_field():
    payload = {k: v for k, v in VALID.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422