"""Regroupement des services Internet en deux indices (scénario C)."""
import pandas as pd

SERVICES_SECURITE = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
SERVICES_STREAMING = ["StreamingTV", "StreamingMovies"]

MAX_SECURITY_INDEX = len(SERVICES_SECURITE)    # 4
MAX_STREAMING_PACK = len(SERVICES_STREAMING)   # 2


def _vers_binaire(serie: pd.Series) -> pd.Series:
    """'Yes' -> 1 ; 'No' et 'No internet service' -> 0. Accepte aussi des colonnes numériques."""
    if pd.api.types.is_numeric_dtype(serie):
        return (serie > 0).astype(int)
    return (serie == "Yes").astype(int)


def regrouper_services(X: pd.DataFrame) -> pd.DataFrame:
    """Ajoute Security_Index (0 à 4) et Streaming_Pack (0 à 2), puis retire les 6 services."""
    X = X.copy()
    X["Security_Index"] = sum(_vers_binaire(X[c]) for c in SERVICES_SECURITE)
    X["Streaming_Pack"] = sum(_vers_binaire(X[c]) for c in SERVICES_STREAMING)
    return X.drop(columns=SERVICES_SECURITE + SERVICES_STREAMING)