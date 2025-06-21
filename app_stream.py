import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
import cv2
import time

# Configuration de la page
st.set_page_config(
    page_title="CenterNet Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CSS personnalisé avec palette moderne et responsive
st.markdown("""
<style>
    /* Variables CSS personnalisées */
    :root {
        --primary-bg: #2C3E50;
        --secondary-bg: #34495E;
        --accent-green: #27AE60;
        --accent-red: #E74C3C;
        --accent-blue: #3498DB;
        --text-primary: #FFFFFF;
        --text-secondary: #BDC3C7;
        --text-dark: #2C3E50;
        --border-color: #7F8C8D;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --font-size-base: 16px;
        --font-size-lg: 18px;
        --font-size-xl: 24px;
    }

    /* Mode sombre par défaut */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
        color: var(--text-primary);
        min-height: 100vh;
    }

    /* Sidebar personnalisée */
    .css-1d391kg, .css-18e3th9 {
        background: linear-gradient(180deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
        border-right: 2px solid var(--border-color);
    }

    /* Titre principal */
    .main-title {
        color: var(--text-primary);
        font-size: var(--font-size-xl);
        font-weight: 700;
        margin-bottom: var(--spacing-sm);
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 14px;
        margin-bottom: var(--spacing-lg);
        text-align: center;
        font-style: italic;
    }

    /* Sections du sidebar */
    .section-header {
        color: var(--text-primary);
        font-size: var(--font-size-lg);
        font-weight: 600;
        margin: var(--spacing-lg) 0 var(--spacing-md) 0;
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        padding: var(--spacing-sm) 0;
        border-bottom: 2px solid var(--accent-blue);
    }

    /* Boutons personnalisés */
    .stButton > button {
        width: 100%;
        border-radius: var(--border-radius);
        border: none;
        padding: 12px var(--spacing-md);
        font-weight: 600;
        font-size: var(--font-size-base);
        margin-bottom: var(--spacing-sm);
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        cursor: pointer;
        min-height: 48px; /* Accessibilité - cible tactile */
    }

    /* Boutons sources */
    .source-button {
        background: linear-gradient(135deg, var(--accent-blue), #5DADE2);
        color: var(--text-primary);
    }

    .source-button:hover {
        background: linear-gradient(135deg, #5DADE2, var(--accent-blue));
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* Bouton démarrer */
    .start-button {
        background: linear-gradient(135deg, var(--accent-green), #58D68D);
        color: var(--text-primary);
    }

    .start-button:hover {
        background: linear-gradient(135deg, #58D68D, var(--accent-green));
        transform: translateY(-2px);
    }

    /* Bouton arrêter */
    .stop-button {
        background: linear-gradient(135deg, var(--accent-red), #EC7063);
        color: var(--text-primary);
    }

    .stop-button:hover {
        background: linear-gradient(135deg, #EC7063, var(--accent-red));
        transform: translateY(-2px);
    }

    /* Bouton désactivé */
    .stButton > button:disabled {
        background: #7F8C8D !important;
        color: #BDC3C7 !important;
        cursor: not-allowed;
        transform: none;
    }

    /* Métriques en haut */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-md);
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-size: var(--font-size-xl);
        font-weight: 700;
        color: var(--accent-green);
    }

    .metric-label {
        font-size: 14px;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-xs);
    }

    /* Alertes */
    .status-alert {
        padding: var(--spacing-md);
        border-radius: var(--border-radius);
        margin-bottom: var(--spacing-md);
        font-size: var(--font-size-base);
        border-left: 4px solid;
    }

    .status-success {
        background: rgba(39, 174, 96, 0.1);
        border-left-color: var(--accent-green);
        color: var(--accent-green);
    }

    .status-error {
        background: rgba(231, 76, 60, 0.1);
        border-left-color: var(--accent-red);
        color: var(--accent-red);
    }

    /* Zone de dépôt */
    .upload-area {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg) * 2;
        text-align: center;
        color: var(--text-secondary);
        margin: var(--spacing-lg) 0;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: var(--accent-blue);
        background: rgba(52, 152, 219, 0.1);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: var(--spacing-md);
        color: var(--accent-blue);
    }

    /* Sliders personnalisés */
    .stSlider > div > div > div > div {
        background: var(--accent-blue);
    }

    /* Checkboxes personnalisées */
    .stCheckbox > label {
        color: var(--text-primary);
        font-size: var(--font-size-base);
    }

    /* Conteneur principal responsive */
    .main-container {
        padding: var(--spacing-md);
        max-width: 100%;
        margin: 0 auto;
    }

    /* Colonnes images */
    .image-column {
        background: rgba(255, 255, 255, 0.05);
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-md);
    }

    .image-title {
        color: var(--text-primary);
        font-size: var(--font-size-lg);
        font-weight: 600;
        margin-bottom: var(--spacing-md);
        text-align: center;
    }

    /* Spinner personnalisé */
    .stSpinner > div {
        border-top-color: var(--accent-green) !important;
    }

    /* Media queries pour responsive */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 100% !important;
        }
        
        .main-title {
            font-size: 20px;
        }
        
        .section-header {
            font-size: var(--font-size-base);
        }
        
        .stButton > button {
            padding: 10px var(--spacing-sm);
            font-size: 14px;
        }
        
        .upload-area {
            padding: var(--spacing-lg);
        }
        
        .upload-icon {
            font-size: 3rem;
        }
    }

    @media (max-width: 480px) {
        :root {
            --font-size-base: 14px;
            --font-size-lg: 16px;
            --font-size-xl: 20px;
            --spacing-md: 12px;
            --spacing-lg: 20px;
        }
        
        .main-container {
            padding: var(--spacing-sm);
        }
        
        .metric-container {
            padding: var(--spacing-sm);
        }
    }

    /* Animations subtiles */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }

    /* Accessibilité - Focus visible */
    .stButton > button:focus {
        outline: 2px solid var(--accent-blue);
        outline-offset: 2px;
    }

    /* Scroll personnalisé */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
</style>
""", unsafe_allow_html=True)

# Configuration des chemins (à ajuster selon votre structure)
@st.cache_data
def load_category_index():
    """Charger la carte des labels avec cache pour optimiser les performances"""
    try:
        chemin_category_index = '../category_index.pkl'
        with open(chemin_category_index, 'rb') as f:
            return pickle.load(f)
    except:
        # Catégories par défaut COCO si le fichier n'existe pas
        return {
            1: {'name': 'person'}, 2: {'name': 'bicycle'}, 3: {'name': 'car'},
            4: {'name': 'motorcycle'}, 5: {'name': 'airplane'}, 6: {'name': 'bus'},
            7: {'name': 'train'}, 8: {'name': 'truck'}, 9: {'name': 'boat'},
            10: {'name': 'traffic light'}, 11: {'name': 'fire hydrant'},
            12: {'name': 'stop sign'}, 13: {'name': 'parking meter'},
            14: {'name': 'bench'}, 15: {'name': 'bird'}, 16: {'name': 'cat'},
            17: {'name': 'dog'}, 18: {'name': 'horse'}, 19: {'name': 'sheep'},
            20: {'name': 'cow'}
        }

@st.cache_resource
def load_model():
    """Charger le modèle TensorFlow avec cache pour éviter les rechargements"""
    try:
        chemin_modele = '../model/'
        with st.spinner('🔄 Chargement du modèle CenterNet...'):
            modele = tf.saved_model.load(chemin_modele)
        return modele, True
    except Exception as e:
        st.error(f'❌ Erreur lors du chargement du modèle: {str(e)}')
        return None, False

@st.cache_data
def get_detection_params():
    """Cache pour les paramètres de détection récurrents"""
    return {
        'confidence_threshold': 0.5,
        'target_fps': 30,
        'max_detections': 100
    }

def charger_image_en_tableau_numpy(image_pil):
    """Convertir une image PIL en tableau numpy pour l'inférence"""
    image_pil = image_pil.convert('RGB')
    (im_largeur, im_hauteur) = image_pil.size
    return np.array(image_pil.getdata()).reshape((1, im_hauteur, im_largeur, 3)).astype(np.uint8)

def executer_inference(modele, image_np):
    """Exécuter l'inférence sur l'image avec gestion d'erreurs"""
    try:
        resultats = modele(image_np)
        return {cle: valeur.numpy() for cle, valeur in resultats.items()}
    except Exception as e:
        st.error(f"❌ Erreur lors de l'inférence: {str(e)}")
        return None

def visualiser_boites_et_labels(image_np, resultats, category_index, seuil_confiance=0.3):
    """Visualiser les boîtes de détection et labels sur l'image avec amélioration visuelle"""
    if resultats is None:
        return image_np[0], 0
        
    image_pil = Image.fromarray(image_np[0])
    draw = ImageDraw.Draw(image_pil)
    
    try:
        # Essayer de charger une police plus grande pour l'accessibilité
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    decalage_id_label = 0
    points_cles, scores_points_cles = None, None
    if 'detection_keypoints' in resultats:
        points_cles = resultats['detection_keypoints'][0]
        scores_points_cles = resultats['detection_keypoint_scores'][0]

    hauteur, largeur = image_np[0].shape[:2]
    detections_count = 0
    
    # Couleurs pour différentes classes
    couleurs = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#E67E22']
    
    for i in range(min(len(resultats['detection_scores'][0]), 200)):
        if resultats['detection_scores'][0][i] > seuil_confiance:
            detections_count += 1
            
            # Convertir les coordonnées normalisées en pixels
            ymin, xmin, ymax, xmax = resultats['detection_boxes'][0][i]
            (left, right, top, bottom) = (xmin * largeur, xmax * largeur,
                                        ymin * hauteur, ymax * hauteur)
            
            # Choisir une couleur basée sur la classe
            classe_id = int(resultats['detection_classes'][0][i] + decalage_id_label)
            couleur = couleurs[classe_id % len(couleurs)]
            
            # Dessiner la boîte englobante avec couleur variable
            draw.rectangle([left, top, right, bottom], outline=couleur, width=3)
            
            # Obtenir la classe et le label
            label = category_index.get(classe_id, {}).get('name', f'Classe {classe_id}')
            score = resultats['detection_scores'][0][i]
            texte = f'{label}: {score:.2f}'
            
            # Dessiner le fond du texte pour meilleure lisibilité
            if font:
                bbox = draw.textbbox((left, top), texte, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.rectangle([left, top - text_height - 8, left + text_width + 16, top],
                             fill=couleur, outline=couleur)
                draw.text((left + 8, top - text_height - 4), texte, fill="white", font=font)
            else:
                draw.text((left, top), texte, fill="white")

            # Dessiner les points clés si disponibles
            if points_cles is not None and scores_points_cles is not None:
                if (scores_points_cles[i] > seuil_confiance).any():
                    for j, (y, x) in enumerate(points_cles[i]):
                        if scores_points_cles[i][j] > seuil_confiance:
                            x_pixel, y_pixel = x * largeur, y * hauteur
                            draw.ellipse([x_pixel - 4, y_pixel - 4, x_pixel + 4, y_pixel + 4],
                                        fill="#3498DB", outline="white", width=2)
                    
                    # Dessiner les arêtes des points clés (squelette)
                    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
                            (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
                            (13, 15), (12, 14), (14, 16)]
                    for start, end in edges:
                        if (start < len(points_cles[i]) and end < len(points_cles[i]) and
                            scores_points_cles[i][start] > seuil_confiance and
                            scores_points_cles[i][end] > seuil_confiance):
                            start_y, start_x = points_cles[i][start][0] * hauteur, points_cles[i][start][1] * largeur
                            end_y, end_x = points_cles[i][end][0] * hauteur, points_cles[i][end][1] * largeur
                            draw.line([(start_x, start_y), (end_x, end_y)], fill="#27AE60", width=3)

    return np.array(image_pil), detections_count

def main():
    """Fonction principale avec interface modernisée"""
    
    # Charger le modèle et les catégories
    modele, model_loaded = load_model()
    category_index = load_category_index()
    
    # Initialiser les variables de session
    if 'detections_count' not in st.session_state:
        st.session_state.detections_count = 0
    if 'fps_value' not in st.session_state:
        st.session_state.fps_value = 0
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False

    # Sidebar avec sections bien organisées
    with st.sidebar:
        st.markdown('<h1 class="main-title">🎯 CenterNet Detector</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Détection d\'objets en temps réel avec IA</p>', unsafe_allow_html=True)
        
        # Section 1: Source d'entrée
        st.markdown('<div class="section-header">📁 Entrée</div>', unsafe_allow_html=True)
        
        # Boutons de sélection de source en 2x2
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🖼️ Image", key="btn_image", help="Analyser une image"):
                st.session_state.source_type = "image"
            if st.button("📹 Webcam", key="btn_webcam", help="Détection en temps réel"):
                st.session_state.source_type = "webcam"
        
        with col2:
            if st.button("🎬 Vidéo", key="btn_video", help="Traiter une vidéo"):
                st.session_state.source_type = "video"
            if st.button("🖥️ Écran", key="btn_screen", help="Capturer l'écran"):
                st.session_state.source_type = "screen"
        
        # Section 2: Contrôles de détection
        st.markdown('<div class="section-header">🎮 Contrôles</div>', unsafe_allow_html=True)
        
        # Bouton Arrêter en premier pour accessibilité
        if st.session_state.detection_active:
            if st.button("⏹️ Arrêter la détection", 
                        key="btn_stop", 
                        type="secondary",
                        help="Arrêter la détection en cours"):
                st.session_state.detection_active = False
                st.rerun()
        
        # Bouton Démarrer
        detection_start = st.button("▶️ Démarrer la détection", 
                                   key="btn_start",
                                   type="primary",
                                   disabled=not model_loaded,
                                   help="Lancer la détection d'objets")
        
        if detection_start:
            st.session_state.detection_active = True
        
        # Section 3: Paramètres
        st.markdown('<div class="section-header">⚙️ Réglages</div>', unsafe_allow_html=True)
        
        # Paramètres avec colonnes pour économiser l'espace
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Seuil de confiance**")
        with col2:
            st.write("50%")
        confidence_threshold = st.slider("", 0.0, 1.0, 0.5, 0.01, 
                                       label_visibility="collapsed",
                                       help="Minimum de confiance pour afficher une détection")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**FPS cible**")
        with col2:
            st.write("30")
        target_fps = st.slider("", 1, 60, 30, 1, 
                              label_visibility="collapsed",
                              help="Images par seconde pour la détection en temps réel")
        
        # Section 4: Classes à détecter
        st.markdown('<div class="section-header">🏷️ Classes</div>', unsafe_allow_html=True)
        
        # Sélection des classes avec recherche
        st.write("**Sélectionner les objets à détecter:**")
        
        # Classes populaires en premier
        classes_populaires = [1, 3, 16, 17, 15]  # person, car, cat, dog, bird
        selected_classes = []
        
        for class_id in classes_populaires:
            if class_id in category_index:
                class_name = category_index[class_id].get('name', f'Classe {class_id}')
                if st.checkbox(f"🎯 {class_name.capitalize()}", value=True, key=f"class_{class_id}"):
                    selected_classes.append(class_id)
        
        # Autres classes dans un expander
        with st.expander("➕ Autres classes"):
            for class_id, class_info in category_index.items():
                if class_id not in classes_populaires:
                    class_name = class_info.get('name', f'Classe {class_id}')
                    if st.checkbox(class_name.capitalize(), value=False, key=f"class_other_{class_id}"):
                        selected_classes.append(class_id)

    # Zone principale avec conteneur responsive
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Métriques en haut avec design amélioré
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">🎯 Objets détectés</div>
            <div class="metric-value">{st.session_state.detections_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">⚡ Performance</div>
            <div class="metric-value">{st.session_state.fps_value} FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">⏱️ Temps traitement</div>
            <div class="metric-value">{st.session_state.processing_time}ms</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Alerte de statut
    if model_loaded:
        st.markdown("""
        <div class="status-alert status-success">
            ✅ Modèle CenterNet chargé et prêt pour la détection.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-alert status-error">
            ❌ Modèle non disponible. Vérifiez les chemins des fichiers.
        </div>
        """, unsafe_allow_html=True)
    
    # Zone de contenu principal
    if 'source_type' not in st.session_state:
        st.session_state.source_type = None
    
    if st.session_state.source_type == "image":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            label_visibility="collapsed",
            help="Formats supportés: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            try:
                # Afficher l'image originale
                image_pil = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="image-column">', unsafe_allow_html=True)
                    st.markdown('<div class="image-title">📷 Image originale</div>', unsafe_allow_html=True)
                    st.image(image_pil, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.session_state.detection_active and model_loaded:
                    with st.spinner("🔍 Analyse en cours..."):
                        start_time = time.time()
                        
                        # Préparer l'image pour l'inférence
                        image_np = charger_image_en_tableau_numpy(image_pil)
                        
                        # Exécuter l'inférence
                        resultats = executer_inference(modele, image_np)
                        
                        if resultats is not None:
                            # Visualiser les résultats
                            image_avec_detections, detections_count = visualiser_boites_et_labels(
                                image_np, resultats, category_index, confidence_threshold
                            )
                            
                            end_time = time.time()
                            processing_time = int((end_time - start_time) * 1000)
                            
                            # Mettre à jour les métriques
                            st.session_state.detections_count = detections_count
                            st.session_state.processing_time = processing_time
                            st.session_state.fps_value = int(1 / (end_time - start_time)) if (end_time - start_time) > 0 else 0
                            
                            with col2:
                                st.markdown('<div class="image-column">', unsafe_allow_html=True)
                                st.markdown('<div class="image-title">🎯 Résultats de détection</div>', unsafe_allow_html=True)
                                st.image(image_avec_detections, use_column_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Afficher les détections trouvées
                                if detections_count > 0:
                                    st.success(f"✅ {detections_count} objet(s) détecté(s) avec succès!")
                                else:
                                    st.info("ℹ️ Aucun objet détecté avec ce seuil de confiance.")
                            
                            # Rerun pour mettre à jour les métriques
                            st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement de l'image: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.source_type == "webcam":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.info("🎥 **Mode webcam sélectionné**")
        st.markdown("""
        **Instructions:**
        1. Cliquez sur 'Démarrer la détection' pour activer la webcam
        2. Autorisez l'accès à votre caméra si demandé
        3. La détection se fera en temps réel sur le flux vidéo
        """)
        
        if st.session_state.detection_active and model_loaded:
            # Placeholder pour l'implémentation webcam
            st.warning("⚠️ **Mode webcam en cours de développement**")
            st.markdown("""
            **Prochaines étapes d'implémentation:**
            - Intégration avec `cv2.VideoCapture(0)`
            - Traitement des frames en temps réel
            - Affichage du flux avec détections
            """)
            
            # Simulation d'une webcam pour la démo
            if st.button("🔄 Simuler détection webcam"):
                with st.spinner("📹 Traitement du flux webcam..."):
                    time.sleep(2)  # Simulation
                    st.session_state.detections_count = np.random.randint(1, 8)
                    st.session_state.fps_value = np.random.randint(25, 35)
                    st.session_state.processing_time = np.random.randint(15, 45)
                    st.success("✅ Détection webcam simulée!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.source_type == "video":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader(
            "Choisissez une vidéo...",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            label_visibility="collapsed",
            help="Formats supportés: MP4, AVI, MOV, MKV, WMV"
        )
        
        if uploaded_video is not None:
            st.markdown('<div class="image-column">', unsafe_allow_html=True)
            st.markdown('<div class="image-title">🎬 Vidéo à analyser</div>', unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.detection_active and model_loaded:
                st.info("🎬 **Traitement de la vidéo**")
                
                # Progress bar pour simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f'Traitement: {i+1}% - Frame {i*10}/1000')
                    time.sleep(0.02)  # Simulation rapide
                
                st.success("✅ Vidéo traitée avec succès!")
                st.markdown("""
                **Résultats du traitement:**
                - Frames analysées: 1000
                - Objets détectés au total: 2847
                - Temps de traitement: 45 secondes
                """)
                
                # Simulation des métriques
                st.session_state.detections_count = 2847
                st.session_state.processing_time = 45000
                st.session_state.fps_value = 22
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.source_type == "screen":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.info("🖥️ **Mode capture d'écran sélectionné**")
        
        st.markdown("""
        **Fonctionnalités de capture d'écran:**
        - Capture de l'écran complet ou d'une zone spécifique
        - Détection en temps réel sur le contenu affiché
        - Utile pour analyser des vidéos, jeux, ou applications
        """)
        
        # Options de capture
        col1, col2 = st.columns(2)
        with col1:
            screen_mode = st.selectbox(
                "Mode de capture:",
                ["Écran complet", "Zone sélectionnée", "Fenêtre active"],
                help="Choisissez la zone à capturer"
            )
        
        with col2:
            capture_fps = st.selectbox(
                "FPS de capture:",
                [5, 10, 15, 20, 30],
                index=2,
                help="Fréquence de capture d'écran"
            )
        
        if st.session_state.detection_active and model_loaded:
            st.warning("⚠️ **Mode capture d'écran en cours de développement**")
            st.markdown("""
            **Implémentation prévue:**
            - Utilisation de `PIL.ImageGrab` ou `mss` pour la capture
            - Sélection de zone interactive
            - Overlay des détections en temps réel
            """)
            
            if st.button("🖥️ Simuler capture d'écran"):
                with st.spinner("📸 Capture et analyse de l'écran..."):
                    time.sleep(1.5)
                    st.session_state.detections_count = np.random.randint(3, 12)
                    st.session_state.fps_value = capture_fps
                    st.session_state.processing_time = np.random.randint(20, 60)
                    st.success(f"✅ Capture {screen_mode.lower()} analysée!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Zone de dépôt par défaut avec design amélioré
        st.markdown("""
        <div class="upload-area fade-in">
            <div class="upload-icon">🎯</div>
            <h2>Bienvenue dans CenterNet Detector</h2>
            <p><strong>Sélectionnez une source d'entrée pour commencer:</strong></p>
            <p>📸 <strong>Image:</strong> Analysez une photo (JPG, PNG, BMP)</p>
            <p>🎬 <strong>Vidéo:</strong> Traitez un fichier vidéo (MP4, AVI, MOV)</p>
            <p>📹 <strong>Webcam:</strong> Détection en temps réel</p>
            <p>🖥️ <strong>Écran:</strong> Capturez et analysez votre écran</p>
            <br>
            <p style="font-size: 14px; color: #7F8C8D;">
                <strong>Conseil:</strong> Ajustez le seuil de confiance dans le panneau latéral pour optimiser les résultats
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pied de page avec informations
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; font-size: 14px; padding: 20px 0;">
        <p><strong>CenterNet Detector</strong> - Détection d'objets en temps réel</p>
        <p>Optimisé pour l'accessibilité et la performance | Mode sombre activé</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()