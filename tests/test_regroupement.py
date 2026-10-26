import pandas as pd

from src.features.regroupement import (
    SERVICES_SECURITE,
    SERVICES_STREAMING,
    regrouper_services,
)


def _exemple():
    return pd.DataFrame({
        "tenure": [10, 20, 30],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
    })


def test_regrouper_services_compte_correctement():
    resultat = regrouper_services(_exemple())
    assert resultat["Security_Index"].tolist() == [3, 0, 0]
    assert resultat["Streaming_Pack"].tolist() == [1, 0, 0]


def test_regrouper_services_supprime_les_six_services():
    resultat = regrouper_services(_exemple())
    for col in SERVICES_SECURITE + SERVICES_STREAMING:
        assert col not in resultat.columns


def test_regrouper_services_conserve_les_autres_colonnes():
    assert "tenure" in regrouper_services(_exemple()).columns


def test_regrouper_services_ne_modifie_pas_l_entree():
    X = _exemple()
    regrouper_services(X)
    assert "Security_Index" not in X.columns
    assert "OnlineSecurity" in X.columns