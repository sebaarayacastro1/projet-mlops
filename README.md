# Projet MLOps

## Guide d'installation et d'exécution

### 1. Prérequis et Dépendances
Installez l'ensemble des bibliothèques nécessaires à l'aide du fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```
*(Dépendances principales : `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `requests`, `httpx`)*

---

### 2. Lancer l'API (FastAPI)
Démarrez le serveur de l'API en mode développement avec Uvicorn :

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
> L'API sera accessible sur `http://localhost:8000` et la documentation interactive (Swagger) sur `http://localhost:8000/docs`.

---

### 3. Lancer l'Interface (Streamlit)
Dans un autre terminal, lancez l'application Streamlit pour interagir avec le modèle :

```bash
streamlit run app.streamlit.py
```
> L'interface s'ouvrira automatiquement dans votre navigateur sur `http://localhost:8501`.

