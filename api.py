"""
API de Détection d'Objets avec CenterNet et FastAPI.

Ce script déploie un modèle de détection d'objets (CenterNet) via une API REST.
Il propose deux endpoints principaux :
1. POST /predict: Envoie une image et reçoit les données de détection (boîtes, classes, scores) au format JSON.
2. POST /predict/image: Envoie une image et reçoit en retour l'image elle-même avec les détections dessinées dessus.

Prérequis d'installation :
pip install "fastapi[all]" tensorflow "pillow>=9.0.0"
"""

import os
import io
import pickle
from typing import Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response

# --- Configuration ---

# Supprimer les avertissements de bas niveau de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Définition des chemins (À MODIFIER PAR L'UTILISATEUR)
CHEMIN_MODELE = '../model/'  # <-- IMPORTANT: Mettez ici le chemin vers votre dossier SavedModel
CHEMIN_CATEGORY_INDEX = 'category_index.pkl'  # <-- IMPORTANT: Mettez ici le chemin vers votre fichier pkl

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Détection d'Objets",
    description="Une API performante pour la détection d'objets avec CenterNet, retournant des données JSON ou des images annotées.",
    version="2.0.0",
)

# Variables globales pour le modèle et les labels (chargés au démarrage)
g = {"model": None, "category_index": None}


# --- Fonctions Utilitaires ---

def charger_modele_et_labels():
    """Charge le modèle TensorFlow et l'index des catégories en mémoire."""
    try:
        print("Chargement du modèle TensorFlow...")
        g["model"] = tf.saved_model.load(CHEMIN_MODELE)
        print("Modèle chargé avec succès.")
        
        print("Chargement de l'index des catégories...")
        with open(CHEMIN_CATEGORY_INDEX, 'rb') as f:
            g["category_index"] = pickle.load(f)
        print("Index des catégories chargé avec succès.")
        
    except FileNotFoundError as e:
        print(f"ERREUR CRITIQUE: Fichier non trouvé - {e}. Assurez-vous que les chemins sont corrects.")
        # Arrêter l'application si les fichiers essentiels ne sont pas trouvés
        raise SystemExit(f"Erreur de démarrage: {e}")
    except Exception as e:
        print(f"ERREUR CRITIQUE: Une erreur inattendue est survenue lors du chargement : {e}")
        raise SystemExit(f"Erreur de démarrage: {e}")


def executer_inference(image_np: np.ndarray) -> Dict[str, np.ndarray]:
    """Exécute l'inférence du modèle sur une image et retourne les résultats."""
    resultats_tensor = g["model"](image_np)
    # Convertit les tenseurs de sortie en tableaux NumPy
    return {key: value.numpy() for key, value in resultats_tensor.items()}


def visualiser_boites_et_labels(
    image_np: np.ndarray,
    resultats: Dict[str, np.ndarray],
    seuil_confiance: float
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Dessine les boîtes englobantes et les labels sur l'image et retourne l'image annotée et les détections."""
    image_pil = Image.fromarray(image_np.squeeze(axis=0))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        # Utilise une police TrueType pour un meilleur rendu visuel
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        # Si la police n'est pas trouvée, utilise la police par défaut
        font = ImageFont.load_default()

    detections = []
    hauteur, largeur = image_pil.height, image_pil.width
    
    scores = resultats['detection_scores'][0]
    boxes = resultats['detection_boxes'][0]
    classes = resultats['detection_classes'][0].astype(np.int64)

    for i in range(scores.shape[0]):
        if scores[i] < seuil_confiance:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin * largeur, xmax * largeur, ymin * hauteur, ymax * hauteur)

        classe_id = classes[i]
        label = g["category_index"].get(classe_id, {}).get('name', f'Classe {classe_id}')
        score = scores[i]
        texte_affichage = f'{label}: {int(score * 100)}%'

        # Dessiner la boîte
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
        
        # Dessiner un fond pour le texte pour une meilleure lisibilité
        text_bbox = draw.textbbox((left, top), texte_affichage, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        draw.rectangle([left, top - text_height - 5, left + text_width + 4, top], fill="red")
        
        # Dessiner le texte du label
        draw.text((left + 2, top - text_height - 3), texte_affichage, fill="white", font=font)

        detections.append({
            "class_id": int(classe_id),
            "class_name": label,
            "confidence": float(score),
            "bbox": [int(left), int(top), int(right), int(bottom)]
        })

    return np.array(image_pil), detections


async def process_image_and_detect(
    file: UploadFile,
    confidence_threshold: float
) -> Dict[str, Any]:
    """
    Fonction centrale qui traite une image uploadée et retourne les résultats de détection.
    Factorise la logique pour éviter la redondance dans les endpoints.
    """
    if g["model"] is None or g["category_index"] is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt. Veuillez patienter.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Type de fichier invalide. Seules les images sont acceptées.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convertir l'image en tableau NumPy
        image_np = np.array(image)[np.newaxis, ...]
        
        # Exécuter l'inférence
        resultats_inference = executer_inference(image_np)
        
        # Visualiser les résultats
        image_annotee_np, detections = visualiser_boites_et_labels(
            image_np, resultats_inference, confidence_threshold
        )
        
        return {
            "annotated_image_np": image_annotee_np,
            "detections": detections,
            "filename": file.filename,
            "image_size": image.size
        }

    except Exception as e:
        print(f"Erreur lors du traitement de l'image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Une erreur interne est survenue: {e}")


# --- Événements et Endpoints de l'API ---

@app.on_event("startup")
async def startup_event():
    """Au démarrage de l'application, charge le modèle et les labels."""
    charger_modele_et_labels()


@app.get("/", summary="Accueil de l'API", include_in_schema=False)
async def root():
    return {"message": "Bienvenue sur l'API de détection d'objets. Consultez /docs pour la documentation."}


@app.get("/health", summary="Vérification de l'état de l'API")
async def health_check():
    """Vérifie si le modèle est chargé et si l'API est fonctionnelle."""
    model_loaded = g["model"] is not None
    return {"status": "healthy" if model_loaded else "loading", "model_loaded": model_loaded}


@app.post(
    "/predict",
    summary="Détecter des objets et retourner les données en JSON",
    tags=["Prediction"]
)
async def predict_json(
    processed_data: dict = Depends(process_image_and_detect)
):
    """
    Uploadez une image pour détecter des objets.
    
    - **Retourne**: Un objet JSON contenant les informations sur les détections.
    """
    return {
        "filename": processed_data["filename"],
        "image_size": {
            "width": processed_data["image_size"][0],
            "height": processed_data["image_size"][1]
        },
        "num_detections": len(processed_data["detections"]),
        "detections": processed_data["detections"]
    }


@app.post(
    "/predict/image",
    summary="Détecter des objets et retourner l'image annotée",
    tags=["Prediction"],
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "L'image annotée est retournée avec succès."}
    }
)
async def predict_image(
    processed_data: dict = Depends(process_image_and_detect)
):
    """
    Uploadez une image pour détecter des objets.
    
    - **Retourne**: L'image au format JPEG avec les boîtes, labels et scores dessinés dessus.
    """
    image_annotee_pil = Image.fromarray(processed_data["annotated_image_np"])
    
    # Sauvegarder l'image dans un buffer en mémoire
    buffer = io.BytesIO()
    image_annotee_pil.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Retourner le buffer comme une réponse en streaming
    return StreamingResponse(buffer, media_type="image/jpeg")


# --- Démarrage de l'application ---

if __name__ == "__main__":
    import uvicorn
    # Lance le serveur Uvicorn. L'option `factory=True` permet de ne pas charger l'app au niveau global
    # ce qui est une bonne pratique, mais pour la simplicité, un appel direct est aussi bien.
    uvicorn.run(app, host="127.0.0.1", port=8000)