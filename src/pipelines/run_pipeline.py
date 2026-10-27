
import subprocess
import sys
from pathlib import Path

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def executer_etape(description, commande):
    """Exécute une étape du pipeline et gère les erreurs."""
    print(f"\n[INFO] {description}...")

    resultat = subprocess.run(
        commande,
        cwd=PROJECT_ROOT,
        check=False
    )

    if resultat.returncode != 0:
        print(f"[ERREUR] Échec de l'étape : {description}")
        sys.exit(resultat.returncode)

    print(f"[SUCCÈS] {description} exécuté avec succès.")


if __name__ == "__main__":
    print("=== DÉMARRAGE DU PIPELINE MLOPS ===")

    # 1. Nettoyage des données
    executer_etape(
        description="Nettoyage des données",
        commande=[sys.executable, "-m", "src.data.clean_data"]
    )

    # 2. Construction des caractéristiques (Features)
    executer_etape(
        description="Ingénierie des caractéristiques",
        commande=[sys.executable, "-m", "src.features.build_features"]
    )

    # 3. Entraînement du modèle
    executer_etape(
        description="Entraînement du modèle",
        commande=[sys.executable, "-m", "src.models.train"]
    )

    # 4. Exécution des tests (Tests unitaires / API)
    executer_etape(
        description="Exécution des tests (pytest)",
        commande=[sys.executable, "-m", "pytest", "-v"]
    )

    print("\n=== PIPELINE EXÉCUTÉ DE BOUT EN BOUT AVEC SUCCÈS ! ===")
