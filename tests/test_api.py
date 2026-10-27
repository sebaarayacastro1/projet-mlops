
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.features.build_features import (
    VARIABLES_CATEGORIELLES,
    VARIABLES_NUMERIQUES,
)

# ============================================================
# Configuration du modèle et de l'API
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


@pytest.fixture(scope="module")
def api_module():
    """Charge l'API uniquement si le modèle existe."""
    if not MODEL_PATH.is_file():
        pytest.fail(
            "model.joblib introuvable — "
            "entraînez le modèle avant d'exécuter les tests API."
        )

    import api.main as api_main
    return api_main


@pytest.fixture(scope="module")
def client(api_module):
    """Crée un client de test pour l'API FastAPI."""
    return TestClient(api_module.app)


# ============================================================
# Données de test
# ============================================================

VALID = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "InternetService": "DSL",
    "Security_Index": 1,
    "Streaming_Pack": 0,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
}


# ============================================================
# Tests de l'API
# ============================================================

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200

    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn"], bool)


def test_predict_invalid_category(client):
    r = client.post(
        "/predict",
        json={**VALID, "Contract": "Weekly"}
    )
    assert r.status_code == 422


def test_predict_missing_field(client):
    payload = {
        k: v for k, v in VALID.items() if k != "tenure"
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_missing_index(client):
    payload = {
        k: v for k, v in VALID.items() if k != "Security_Index"
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_security_index_out_of_range(client):
    r = client.post(
        "/predict",
        json={**VALID, "Security_Index": 5}
    )
    assert r.status_code == 422


def test_predict_streaming_pack_out_of_range(client):
    r = client.post(
        "/predict",
        json={**VALID, "Streaming_Pack": 3}
    )
    assert r.status_code == 422


def test_predict_indices_without_internet_rejected(client):
    payload = {
        **VALID,
        "InternetService": "No",
        "Security_Index": 2
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_without_internet_with_zero_indices(client):
    payload = {
        **VALID,
        "InternetService": "No",
        "Security_Index": 0,
        "Streaming_Pack": 0
    }
    assert client.post("/predict", json=payload).status_code == 200


# ============================================================
# Cohérence entre les champs de l'API et le modèle
# ============================================================

def test_champs_api_et_variables_du_modele_sont_identiques(api_module):
    champs_api = set(api_module.Customer.model_fields)

    variables_modele = set(
        VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES
    )

    assert champs_api == variables_modele, (
        f"Champs de l'API non utilisés par le modèle : "
        f"{champs_api - variables_modele} | "
        f"Variables du modèle absentes de l'API : "
        f"{variables_modele - champs_api}"
    )
