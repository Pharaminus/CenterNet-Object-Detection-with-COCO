# CenterNet Detector Pro 🚀

![Version](https://img.shields.io/badge/version-2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Système avancé de détection d'objets utilisant CenterNet avec API REST FastAPI et interface web Streamlit moderne

## 📋 Table des matières

- [Présentation](#-présentation)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Interface Streamlit](#-interface-streamlit)
- [Configuration](#-configuration)
- [Dépannage](#-dépannage)
- [Contribution](#-contribution)
- [License](#-license)

## 🎯 Présentation

CenterNet Detector Pro est un système de détection d'objets haute performance basé sur le modèle CenterNet et TensorFlow 2.x. Il offre une solution complète pour la détection d'objets en temps réel avec :

- **API REST FastAPI** pour l'intégration programmatique
- **Interface Streamlit** avec un thème moderne bleu nuit
- **Support multi-sources** : images, vidéos, webcam, capture d'écran
- **Métriques temps réel** et visualisations avancées

## ✨ Fonctionnalités

### 🔧 API FastAPI
- ✅ **Endpoints REST** pour intégration programmatique
- ✅ **Formats supportés** : JPG, PNG, BMP, TIFF, WebP
- ✅ **Réponses JSON** ou images annotées
- ✅ **Gestion d'erreurs** avancée
- ✅ **Documentation Swagger** intégrée

### 🎨 Interface Streamlit
- ✅ **Thème moderne** bleu nuit avec animations
- ✅ **Sources multiples** : images, vidéos, webcam, écran
- ✅ **Métriques temps réel** : FPS, objets détectés, temps de traitement
- ✅ **Contrôles avancés** : seuil de confiance, classes personnalisées
- ✅ **Visualisations** modernes avec boîtes englobantes colorées

### 🎯 Détection
- ✅ **80 classes COCO** supportées
- ✅ **Profils prédéfinis** : Personnes & Véhicules, Animaux, Nourriture
- ✅ **Seuil de confiance** ajustable
- ✅ **Performance optimisée** avec TensorFlow SavedModel

## 🛠 Installation

### Prérequis
- Python 3.8+
- 8 Go RAM minimum
- GPU recommandé pour de meilleures performances

### Installation rapide

```bash
# Cloner le repository
git clone <url-du-dépôt>
cd centerNet-detector-pro

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```txt
fastapi[all]
tensorflow>=2.0.0
pillow>=9.0.0
streamlit>=1.0.0
opencv-python
numpy
```

### Configuration du modèle

```bash
# Structure requise
├── api.py
├── app.py
├── ../model/               # Modèle TensorFlow SavedModel
├── category_index.pkl      # Mapping des classes
└── requirements.txt
```

## 🚀 Usage

### Lancer l'API

```bash
python api.py
```

📍 **Accès** : http://127.0.0.1:8000  
📚 **Documentation** : http://127.0.0.1:8000/docs

### Lancer l'interface Streamlit

```bash
streamlit run app.py
```

📍 **Accès** : http://localhost:8501

## 📡 API Documentation

### Endpoints disponibles

#### `POST /predict`
Détection d'objets avec réponse JSON

```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/predict
```

**Réponse :**
```json
{
  "filename": "image.jpg",
  "image_size": [640, 480],
  "num_detections": 3,
  "detections": [
    {
      "class": "person",
      "score": 0.95,
      "bbox": [100, 50, 200, 300]
    }
  ]
}
```

#### `POST /predict/image`
Détection avec image annotée

```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/predict/image > output.jpg
```

#### `GET /health`
Vérification de l'état de l'API

```bash
curl http://127.0.0.1:8000/health
```

## 🎨 Interface Streamlit

### Fonctionnalités principales

1. **Sources d'entrée**
   - 📸 Upload d'images
   - 🎥 Traitement vidéo frame par frame
   - 📹 Webcam temps réel (simulé)
   - 🖥️ Capture d'écran (simulé)

2. **Contrôles avancés**
   - 🎯 Seuil de confiance (0.0 - 1.0)
   - ⚡ FPS cible (1 - 60)
   - 📊 Nombre max de détections (10 - 500)
   - 🏷️ Sélection de classes personnalisée

3. **Métriques temps réel**
   - 📈 Objets détectés
   - ⏱️ FPS actuel
   - 🔄 Temps de traitement

### Profils de classes prédéfinis

| Profil | Classes incluses |
|--------|------------------|
| 🚗 **Personnes & Véhicules** | person, car, truck, bus, motorcycle, bicycle |
| 🐕 **Animaux** | dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe |
| 🍕 **Nourriture** | banana, apple, sandwich, orange, pizza, donut, cake |
| 🏠 **Objets du quotidien** | chair, sofa, bed, dining table, toilet, tv, laptop, mouse, keyboard |

## ⚙️ Configuration

### API (api.py)
```python
CHEMIN_MODELE = "../model/"  # Chemin vers le SavedModel
CHEMIN_CATEGORY_INDEX = "category_index.pkl"  # Mapping des classes
SEUIL_CONFIANCE_DEFAUT = 0.3  # Seuil par défaut
```

### Streamlit (app.py)
```python
chemin_modele = "../model/"
chemin_category_index = "category_index.pkl"
# Paramètres configurables via l'interface
```

## 🔧 Dépannage

### Erreurs communes

#### Modèle non trouvé
```bash
# Vérifier les chemins
ls ../model/
ls category_index.pkl
```

#### Problèmes de dépendances
```bash
# Réinstaller dans un environnement propre
pip install --upgrade pip
pip install -r requirements.txt
```

#### Performances lentes
- ✅ Activer l'accélération GPU
- ✅ Réduire la résolution d'entrée
- ✅ Ajuster le nombre max de détections

#### Problèmes GPU
```bash
# Vérifier TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📈 Performances

| Métrique | CPU | GPU |
|----------|-----|-----|
| **FPS** | 5-10 | 30-60 |
| **Latence** | 100-200ms | 15-30ms |
| **Mémoire** | 2-4 GB | 4-8 GB |

## 🤝 Equipe

- BALEKAMEN BABATACK LANDRY
- MBASSI ATANGANA YANNICK SERGE 
- VOUKENG TEDONKEMWA ERDI DÉSIRÉ 
- WATAT YONDEP STIVE KEVIN


## Depot git
`https://github.com/Pharaminus/CenterNet-Object-Detection-with-COCO`

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

**CenterNet Detector Pro v2.0** - *Thème Bleu Nuit Premium*

Développé avec ❤️ pour la communauté de détection d'objets

[⭐ Star ce repo](../../stargazers) | [🐛 Report Bug](../../issues) | [✨ Request Feature](../../issues)

</div>