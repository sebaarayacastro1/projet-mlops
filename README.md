# Prédiction du churn client — Projet MLOps

## Présentation

Ce projet a pour objectif de prédire le risque de résiliation des clients d'un opérateur de télécommunications (*Customer Churn*).

Il met en œuvre une architecture MLOps intégrant la préparation des données, l'entraînement de modèles de Machine Learning, le suivi des expériences, une API de prédiction, une interface web, des tests automatisés et un déploiement avec Docker.

## Données

Le projet utilise le jeu de données public **Telco Customer Churn**.

- **7 043 clients** ; **21 colonnes** dans le jeu de données initial.
- **26,5 %** des clients résilient leur abonnement.
- **11 variables** sont utilisées pour la prédiction finale.

Les données comprennent des informations sur le profil des clients, leur ancienneté, leurs abonnements, leurs services et leur facturation.

## Architecture MLOps

Le projet repose sur les composants suivants :

- **Python et scikit-learn** : préparation des données et entraînement des modèles.
- **MLflow** : suivi des expériences et enregistrement des métriques.
- **FastAPI** : exposition des prédictions via une API REST.
- **Streamlit** : interface web permettant de saisir les informations d'un client.
- **Docker Compose** : lancement des services conteneurisés.
- **Pytest** : tests automatisés.
- **GitHub Actions** : intégration continue.

### Pipeline de Machine Learning

```text
Données brutes
      |
      v
Nettoyage des données
      |
      v
Préparation et transformation des variables
      |
      v
Entraînement et comparaison des modèles (MLflow)
      |
      v
Sélection et sauvegarde du meilleur modèle
      |
      v
API FastAPI
      |
      v
Interface Streamlit
```

## Modélisation

Trois algorithmes sont comparés :

1. Régression logistique
2. Random Forest
3. XGBoost

Le modèle retenu est une **régression logistique avec `class_weight="balanced"`**.

La préparation des variables utilise notamment `StandardScaler`, `OneHotEncoder` et deux indices regroupant les services Internet (`Security_Index` et `Streaming_Pack`).

## Résultats

Performances du modèle retenu sur le jeu de test, avec un seuil de décision de 0,5 :

| Métrique | Valeur |
|---|---:|
| ROC-AUC | 0,8368 |
| Précision | 0,4983 |
| Rappel | 0,7861 |
| F1-score | 0,6100 |

Le modèle identifie environ **79 % des clients** qui résilient effectivement leur abonnement.

Le seuil de décision par défaut est **0,5**. Il peut être configuré avec la variable d'environnement `CHURN_THRESHOLD` (valeur comprise entre 0 et 1).

**Attention :** les scores fournis par le modèle ne sont pas calibrés. Ils représentent des **scores de risque**, et non des probabilités réelles de résiliation.

## Démarrage du projet

Deux modes d'exécution sont proposés. **Choisissez l'une des deux méthodes** selon votre besoin.

### Option A — Exécution avec Docker (recommandée pour démarrer les services)

**Prérequis :** Docker et Docker Compose installés et opérationnels.

Depuis la racine du projet :

```bash
docker compose up --build
```

Docker Compose construit et lance les deux services définis dans `docker-compose.yml` :

- **API FastAPI** : http://localhost:8000/docs
- **Interface Streamlit** : http://localhost:8501
- **Vérification de l'API** : http://localhost:8000/health

Pour lancer les services en arrière-plan :

```bash
docker compose up --build -d
```

Pour vérifier les conteneurs et consulter leurs journaux :

```bash
docker compose ps
docker compose logs -f
```

Pour arrêter les services :

```bash
docker compose down
```

**Modèle entraîné :** l'API a besoin de `models/model.joblib` au démarrage. La procédure de génération du modèle doit être assurée par le `Dockerfile` de l'API (si le pipeline y est exécuté à la construction) ou réalisée avant le lancement de Docker si votre configuration utilise un modèle généré localement. Vérifiez le comportement du `Dockerfile` de votre dépôt ; l'absence du modèle empêche l'API de démarrer.

### Option B — Exécution en local (développement)

**Prérequis :** Python 3.11 et pip. Toutes les commandes suivantes sont lancées depuis la racine du projet, sauf indication contraire.

**1. Créer et activer un environnement virtuel**

```bash
python -m venv .venv
```

Sous Windows (PowerShell) :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sous Linux / macOS :

```bash
source .venv/bin/activate
```

**2. Installer les dépendances**

```bash
python -m pip install -r requirements.txt
```

Installer également les dépendances de l'interface Streamlit, définies dans le fichier `requirements.streamlit.txt` :

```bash
python -m pip install -r requirements.streamlit.txt
```

**3. Générer les données traitées et le modèle**

```bash
python -m src.pipelines.run_pipeline
```

Ce pipeline exécute le nettoyage, la préparation des variables, l'entraînement et les tests. Il génère notamment le fichier `models/model.joblib`, nécessaire à l'API.

**4. Démarrer l'API FastAPI** (terminal 1, environnement virtuel actif)

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

L'API et sa documentation sont accessibles aux adresses suivantes :

- Documentation interactive : http://localhost:8000/docs
- Vérification de l'état : http://localhost:8000/health

**5. Démarrer l'interface Streamlit** (terminal 2, même environnement virtuel actif)

```bash
python -m streamlit run app/streamlit_app.py
```

Ouvrir http://localhost:8501 puis remplir le formulaire et cliquer sur **Prédire**. Par défaut, l'interface locale interroge l'API à l'adresse `http://127.0.0.1:8000/predict`.

**6. Consulter les expériences MLflow** (facultatif, terminal 3)

```bash
python -m mlflow ui
```

Ouvrir http://localhost:5000.

### Configuration du seuil de décision

La variable d'environnement `CHURN_THRESHOLD` définit le seuil utilisé pour classer les clients à risque. La valeur par défaut est `0.5`.

Avec Docker, le fichier `docker-compose.yml` comporte :

```yaml
environment:
  CHURN_THRESHOLD: ${CHURN_THRESHOLD:-0.5}
```

Pour utiliser un autre seuil, définir `CHURN_THRESHOLD` dans l'environnement (ou dans un fichier `.env` lu par Docker Compose), puis **recréer le conteneur de l'API**. Par exemple, dans un fichier `.env` local non versionné :

```dotenv
CHURN_THRESHOLD=0.34
```

En local, définir cette variable **avant de démarrer** FastAPI. Sans configuration particulière, le seuil `0.5` reste utilisé.

## Tests et intégration continue

Les tests automatisés couvrent notamment le nettoyage des données, le regroupement des services, les prédictions et les validations de l'API.

Pour exécuter les tests après génération du modèle :

```bash
python -m pytest -v
```

Pour mesurer également la couverture des sources :

```bash
python -m pytest -v --cov=src
```

Le workflow GitHub Actions exécute des contrôles lors des modifications des branches `develop` et `main`.

## Structure du projet

```text
projet-mlops/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipelines/
├── tests/
├── .github/workflows/
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Les données traitées et les fichiers du modèle sont générés par le pipeline et ne sont normalement pas versionnés.

## Limites et perspectives

- Utilisation d'un jeu de données public et statique.
- Absence de données réelles sur le coût des actions de rétention.
- Scores de risque non calibrés.
- Absence de surveillance de la dérive des données et des performances en production.
- Évaluation finale sur un seul jeu de test.

Les évolutions possibles comprennent la calibration des probabilités, le monitoring du modèle et le suivi des performances dans le temps.

## Conclusion

Ce projet illustre une chaîne MLOps reproductible allant de la préparation des données à l'utilisation d'un modèle de prédiction via une API et une interface web. Il combine modélisation, suivi des expériences, automatisation des tests et conteneurisation afin de faciliter la maintenance et l'évolution du système.
