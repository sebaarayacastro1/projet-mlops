# Projet MLOps

## Guide d'installation et d'exécution

### Option 1 : Exécution avec Docker (Recommandée)

#### Prérequis
Docker Desktop doit être installé et en cours d'exécution sur votre machine.

#### Démarrage des services
Pour construire les images et démarrer simultanément l'API et l'interface Streamlit :

```bash
docker-compose up --build
```

Les services seront accessibles aux adresses suivantes :
- Documentation de l'API (Swagger UI) : http://localhost:8000/docs
- Interface utilisateur (Streamlit) : http://localhost:8501

Pour arrêter les conteneurs, utilisez Ctrl + C puis exécutez :
```bash
docker-compose down
```

---

### Option 2 : Exécution en local (Sans Docker)

#### 1. Prérequis et Dépendances
Installez l'ensemble des bibliothèques nécessaires à l'aide du fichier requirements.txt :

```bash
pip install -r requirements.txt
```
*(Dépendances principales : fastapi, uvicorn, pydantic, streamlit, requests, httpx)*

#### 2. Lancer l'API (FastAPI)
Démarrez le serveur de l'API en mode développement avec Uvicorn :

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
L'API sera accessible sur http://localhost:8000 et la documentation interactive (Swagger) sur http://localhost:8000/docs.

#### 3. Lancer l'Interface (Streamlit)
Dans un autre terminal, lancez l'application Streamlit :

```bash
streamlit run app/streamlit_app.py
```
L'interface s'ouvrira automatiquement dans votre navigateur sur http://localhost:8501.