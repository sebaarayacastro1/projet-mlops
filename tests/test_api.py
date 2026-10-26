from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID = {
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "InternetService": "DSL",
    "Security_Index": 1, "Streaming_Pack": 0,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85
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


def test_predict_missing_index():
    payload = {k: v for k, v in VALID.items() if k != "Security_Index"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_security_index_out_of_range():
    r = client.post("/predict", json={**VALID, "Security_Index": 5})
    assert r.status_code == 422


def test_predict_streaming_pack_out_of_range():
    r = client.post("/predict", json={**VALID, "Streaming_Pack": 3})
    assert r.status_code == 422


def test_predict_indices_without_internet_rejected():
    payload = {**VALID, "InternetService": "No", "Security_Index": 2}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_without_internet_with_zero_indices():
    payload = {**VALID, "InternetService": "No", "Security_Index": 0, "Streaming_Pack": 0}
    assert client.post("/predict", json=payload).status_code == 200