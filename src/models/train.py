from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =============================================================
# Chemins du projet
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

NOM_EXPERIENCE = "churn_prediction"


def charger_donnees():
    """Charge les ensembles d'entraînement/test et le préprocesseur déjà entraînés (Phase 2)."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    preprocesseur = joblib.load(PREPROCESSOR_PATH)
    return X_train, X_test, y_train, y_test, preprocesseur


def obtenir_modeles() -> dict:
    """Définit les algorithmes candidats à comparer."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=200, random_state=42
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=3,  # compense le déséquilibre ~73/27 (No/Yes)
            eval_metric="logloss",
            random_state=42,
        ),
    }


def entrainer_et_logger(
    nom_modele: str,
    classifieur,
    preprocesseur,
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[str, float, Pipeline]:
    """Entraîne un modèle, log ses métriques/artefacts dans MLflow et renvoie son ROC-AUC."""
    with mlflow.start_run(run_name=nom_modele):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocesseur),
                ("classifier", classifieur),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        rappel = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Paramètres et métriques
        mlflow.log_param("model_type", nom_modele)
        mlflow.log_params(classifieur.get_params())
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", rappel)
        mlflow.log_metric("f1_score", f1)

        # Matrice de confusion en tant qu'artefact visuel
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        ax.set_title(f"Matrice de confusion - {nom_modele}")
        mlflow.log_figure(fig, f"confusion_matrix_{nom_modele}.png")
        plt.close(fig)

        # Modèle complet (préprocesseur + classifieur)
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        print(f"[{nom_modele}] ROC-AUC={auc:.4f} | Precision={precision:.4f} | "
              f"Recall={rappel:.4f} | F1={f1:.4f}")

        return nom_modele, auc, pipeline


def main():
    mlflow.set_experiment(NOM_EXPERIENCE)

    X_train, X_test, y_train, y_test, preprocesseur = charger_donnees()
    modeles = obtenir_modeles()

    resultats = []
    for nom_modele, classifieur in modeles.items():
        resultats.append(
            entrainer_et_logger(
                nom_modele, classifieur, preprocesseur, X_train, X_test, y_train, y_test
            )
        )

    # Sélection du meilleur modèle selon le ROC-AUC
    meilleur_nom, meilleur_auc, meilleur_pipeline = max(resultats, key=lambda r: r[1])
    print(f"\nMeilleur modèle : {meilleur_nom} (ROC-AUC={meilleur_auc:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meilleur_pipeline, MODEL_PATH)
    print(f"Modèle gagnant enregistré dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()