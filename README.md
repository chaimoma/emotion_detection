# Détection d'Émotions Faciales (CNN + OpenCV + FastAPI + PostgreSQL)

Ce projet est un prototype d'API d'IA, réalisé pour un brief de Simplon. L'objectif est de fournir un service web capable de recevoir une image, de détecter un visage, de prédire l'émotion correspondante, et d'enregistrer ce résultat dans une base de données PostgreSQL.

## 🛠️ Technologies Principales
* **Modèle IA :** TensorFlow / Keras
* **Détection Faciale :** OpenCV (Haar Cascade)
* **API :** FastAPI, Uvicorn
* **Base de Données :** PostgreSQL, SQLAlchemy
* **Tests :** Pytest
* **CI/CD :** GitHub Actions

---

## 🧠 Étape 1 : Le Pipeline Machine Learning (ML)

La première partie du projet a consisté à créer et tester le pipeline d'IA.

### 1. Entraînement du Modèle CNN
L'entraînement du modèle de classification d'émotions a été réalisé dans le notebook `notebook/cnn_training.ipynb`.

* **Google Colab :** Pour bénéficier de la puissance de calcul d'un GPU et accélérer l'entraînement, le notebook a été exécuté sur Google Colab.
* **Dataset :** Le modèle a été entraîné sur un jeu de données de 7 émotions (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`).
* **Bonus :** La "Data Augmentation" (rotation, zoom et flip aléatoires) a été utilisée pour améliorer la précision du modèle et réduire le surapprentissage.
* **Résultat :** Le modèle final a été sauvegardé dans `saved_model/emotion_model.h5`.

### 2. Détection de Visage (OpenCV)
Pour que le CNN puisse se concentrer uniquement sur le visage, un détecteur de visage classique a été utilisé.

* **Outil :** `haarcascade_frontalface_default.xml` d'OpenCV.
* **Rôle :** Il scanne l'image et renvoie les coordonnées `(x, y, w, h)` du visage.

### 3. Script de Test (`detect_and_predict.py`)
Pour valider le pipeline avant de construire l'API, le script `detect_and_predict.py` a été créé. Ce script :
1.  Charge le modèle CNN (`.h5`) et le Haar Cascade (`.xml`).
2.  Lit une image de test.
3.  Utilise le Haar Cascade pour **trouver** le visage.
4.  Recadre et redimensionne le visage en 48x48 pixels.
5.  Utilise le CNN pour **prédire** l'émotion.
6.  Affiche le résultat avec un rectangle et une étiquette sur l'image.

---

## 🖥️ Étape 2 : Le Backend (API et Base de Données)

Une fois le pipeline ML validé, il a été intégré dans une API web.

### 1. API FastAPI (`app/main.py`)
Le cœur du backend est une application FastAPI qui expose deux "endpoints" (routes) :

* **`POST /predict_emotion`**:
    1.  Reçoit une image de l'utilisateur (`UploadFile`).
    2.  Exécute le pipeline ML (détection + prédiction) sur l'image.
    3.  Sauvegarde le résultat (`id`, `emotion`, `confidence`, `created_at`) dans la base de données.
    4.  Retourne le résultat au format JSON.

* **`GET /history`**:
    1.  Se connecte à la base de données.
    2.  Récupère l'historique de toutes les prédictions précédentes.
    3.  Retourne l'historique au format JSON.

### 2. Base de Données (`PostgreSQL`)
Les résultats sont stockés dans une base de données PostgreSQL, gérée avec SQLAlchemy.
* **`app/database.py`**: Gère la connexion à la base de données.
* **`app/models.py`**: Définit la structure de la table `predictions`.
* **`.env`**: Les identifiants de la base de données sont chargés à partir d'un fichier `.env` (qui est ignoré par Git) pour plus de sécurité.

---

## 🚀 Installation et Lancement

Instructions pour lancer le projet localement sur Ubuntu.

### 1. Cloner le Dépôt
```
git clone [https://github.com/chaimoma/emotion_detection.git](https://github.com/chaimoma/emotion_detection.git)
cd emotion_detection
```
### 2. Environnement Virtuel et Dépendances
# Créer et activer l'environnement
```
python3 -m venv venv
source venv/bin/activate
```
# Installer les librairies
```
pip install -r requirements.txt
```
##  Configuration de la Base de Données (PostgreSQL)
#installer postgreSQL (sur ubuntu)
```
sudo apt update
sudo apt install postgresql
```
#Créer l'utilisateur et la base de données :
# Se connecter en tant qu'utilisateur admin postgres
```
sudo -i -u postgres
psql
```
# Exécuter ces commandes SQL (choisissez un mot de passe sécurisé)
```
CREATE DATABASE emotion_api_db;
CREATE USER emotion_user WITH PASSWORD 'votre_mot_de_passe_secret';
GRANT ALL PRIVILEGES ON DATABASE emotion_api_db TO emotion_user;
GRANT USAGE, CREATE ON SCHEMA public TO emotion_user;
\q
exit
```
#Créer le fichier .env : Copiez le fichier d'exemple et modifiez-le avec vos identifiants.
```
cp .env.example .env
nano .env
```
Votre fichier .env doit ressembler à ceci :
```
USER_DB=emotion_user
USER_DB_PASSWORD=votre_mot_de_passe_secret
DB_HOST=localhost
DB_NAME=emotion_api_db
PORT=5432
```
### 5. Lancer le Serveur API
# Lancer l'API avec Uvicorn
```
uvicorn app.main:app --reload
```
Ouvrez votre navigateur à l'adresse http://127.0.0.1:8000/docs pour tester l'API.

🧪 Tests et CI/CD
Tests Unitaires

Les tests sont séparés pour valider les différentes parties du brief :

    tests/test_model.py: Valide que le modèle (.h5) se charge correctement (Exigence 1).

    tests/test_format.py: Valide que l'API (/history) se connecte à la BDD et renvoie le bon format JSON (Exigence 2).

# Lancer les tests

python -m pytest tests/test_model.py
python -m pytest tests/test_format.py

# GitHub Actions

Le fichier .github/workflows/ci.yml automatise les tests. Pour garantir un test rapide et stable, le workflow ne lance que le test_model.py, qui valide l'exigence principale du brief d
