import subprocess
import sys

def executer_etape(description, commande):
    """Exécute une étape du pipeline et gère les erreurs."""
    print(f"\n[INFO] {description}...")
    resultat = subprocess.run(commande, shell=True)
    if resultat.returncode != 0:
        print(f"[ERREUR] Échec de l'étape : {description}")
        sys.exit(resultat.returncode)
    print(f"[SUCCÈS] {description} exécuté avec succès.")

if __name__ == "__main__":
    print("=== DÉMARRAGE DU PIPELINE MLOPS ===")
    
    # 1. Nettoyage des données
    executer_etape(
        description="Nettoyage des données",
        commande="python -m src.data.clean_data"
    )
    
    # 2. Construction des caractéristiques (Features)
    executer_etape(
        description="Ingénierie des caractéristiques",
        commande="python -m src.features.build_features"
    )
    
    # 3. Entraînement du modèle
    executer_etape(
        description="Entraînement du modèle",
        commande="python -m src.models.train"
    )
    
    # 4. Exécution des tests (Tests unitaires / API)
    executer_etape(
        description="Exécution des tests (pytest)",
        commande="pytest -v"
    )
    
    print("\n=== PIPELINE EXÉCUTÉ DE BOUT EN BOUT AVEC SUCCÈS ! ===")