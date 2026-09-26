from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VARIABLES_NUMERIQUES = ['tenure', 'MonthlyCharges', 'TotalCharges']
VARIABLES_CATEGORIELLES = ['gender', 'Partner', 'Dependents', 'PhoneService',
                            'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']


def split_data(df: pd.DataFrame):
    """Sépare les données en ensembles d'entraînement et de test."""
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    """Construit le pipeline de prétraitement des variables."""
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), VARIABLES_NUMERIQUES),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), VARIABLES_CATEGORIELLES)
    ])


def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Préprocesseur enregistré dans : {PREPROCESSOR_PATH}")

    # Sauvegarde des ensembles train/test pour la Phase 3
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print("Ensembles d'entraînement et de test enregistrés.")


if __name__ == "__main__":
    main()