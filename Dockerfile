# 1. Utiliser une image officielle Python comme base
FROM python:3.10-slim

# 2. Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# 3. Copier le fichier de dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier l'ensemble du code source du projet dans le conteneur
COPY . .

# 5. Exposer le port sur lequel s'exécute l'API FastAPI
EXPOSE 8000

# 6. Commande pour démarrer l'API FastAPI avec Uvicorn au lancement du conteneur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]