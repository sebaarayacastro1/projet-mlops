from pathlib import Path
import pandas as pd

# Définir automatiquement les chemins relatifs à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Charge les données brutes depuis le répertoire local."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Le fichier de données n'a pas été trouvé à l'emplacement : {file_path}"
        )
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et transforme le DataFrame Telco Churn."""
    df_clean = df.copy()

    # 1. Supprimer la colonne ID car elle n'est pas prédictive
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 2. Corriger les espaces vides dans TotalCharges et convertir en numérique
    df_clean["TotalCharges"] = df_clean["TotalCharges"].replace(
        r"^\s*$", None, regex=True
    )
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )

    # 3. Remplacer les valeurs manquantes de TotalCharges
    #    (clients avec tenure == 0)
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # 4. Convertir la variable cible 'Churn' au format binaire (1 et 0)
    if "Churn" in df_clean.columns and df_clean["Churn"].dtype == "object":
        df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    return df_clean


def save_data(
    df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH
) -> None:
    """Enregistre le DataFrame traité dans le dossier data/processed/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Données nettoyées enregistrées dans : {output_path}")


def main():
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    save_data(clean_df)


if __name__ == "__main__":
    main()