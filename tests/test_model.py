from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


@pytest.fixture(scope="module")
def modele_entraine():
    if not MODEL_PATH.exists():
        pytest.skip("model.joblib introuvable — exécutez d'abord src/models/train.py")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def echantillon_valide():
    """Un seul client au format attendu par le pipeline (avant préprocessing)."""
    return pd.DataFrame({
        "gender": ["Female"], "SeniorCitizen": [0], "Partner": ["Yes"],
        "Dependents": ["No"], "tenure": [12], "PhoneService": ["Yes"],
        "MultipleLines": ["No"], "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"], "OnlineBackup": ["No"],
        "DeviceProtection": ["No"], "TechSupport": ["No"],
        "StreamingTV": ["No"], "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"], "MonthlyCharges": [55.5],
        "TotalCharges": [666.0],
    })


def test_predict_accepte_un_dataframe_valide(modele_entraine, echantillon_valide):
    prediction = modele_entraine.predict(echantillon_valide)
    assert prediction.shape == (1,)


def test_predict_proba_renvoie_des_probabilites_valides(modele_entraine, echantillon_valide):
    probabilites = modele_entraine.predict_proba(echantillon_valide)
    assert probabilites.shape == (1, 2)
    assert (probabilites >= 0).all() and (probabilites <= 1).all()
    # La somme des probabilités de chaque ligne doit être ~1
    assert probabilites.sum(axis=1)[0] == pytest.approx(1.0, abs=1e-6)


def test_predict_proba_classe_positive_est_un_scalaire_valide(modele_entraine, echantillon_valide):
    proba_churn = modele_entraine.predict_proba(echantillon_valide)[:, 1][0]
    assert 0.0 <= proba_churn <= 1.0