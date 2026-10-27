import pandas as pd
import pytest
from src.data.clean_data import clean_data


@pytest.fixture
def donnees_brutes_exemple():
    """Crée un petit DataFrame simulant les données brutes, avec un cas TotalCharges vide."""
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB"],
        "gender": ["Female", "Male"],
        "tenure": [0, 12],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [" ", "840.0"],  # espace vide simulant un nouveau client
        "Churn": ["No", "Yes"],
    })


def test_clean_data_supprime_customer_id(donnees_brutes_exemple):
    df_propre = clean_data(donnees_brutes_exemple)
    assert "customerID" not in df_propre.columns


def test_clean_data_ne_change_pas_le_nombre_de_lignes(donnees_brutes_exemple):
    df_propre = clean_data(donnees_brutes_exemple)
    assert len(df_propre) == len(donnees_brutes_exemple)


def test_clean_data_gere_totalcharges_vide(donnees_brutes_exemple):
    df_propre = clean_data(donnees_brutes_exemple)
    assert df_propre["TotalCharges"].isnull().sum() == 0
    assert df_propre.loc[0, "TotalCharges"] == 0.0


def test_clean_data_convertit_totalcharges_en_numerique(donnees_brutes_exemple):
    df_propre = clean_data(donnees_brutes_exemple)
    assert pd.api.types.is_numeric_dtype(df_propre["TotalCharges"])


def test_clean_data_encode_churn_en_binaire(donnees_brutes_exemple):
    df_propre = clean_data(donnees_brutes_exemple)
    assert set(df_propre["Churn"].unique()) <= {0, 1}
    assert df_propre.loc[1, "Churn"] == 1  # "Yes" -> 1