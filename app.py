import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# Essayer d'importer TensorFlow avec gestion d'erreur
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f" TensorFlow non disponible: {e}")
    TENSORFLOW_AVAILABLE = False

import shutil

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malaria_secret_key_prod")

# Configuration pour la production
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Limiter la taille des uploads (16MB max)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fonction pour télécharger automatiquement le modèle
def download_model():
    """Télécharge le modèle depuis Google Drive si absent"""
    if not TENSORFLOW_AVAILABLE:
        print(" TensorFlow non disponible, impossible de charger le modèle")
        return False
        
    MODEL_PATH = "best_model.h5"
    
    if os.path.exists(MODEL_PATH):
        print(" Modèle déjà présent")
        return True
        
    print("📥 Téléchargement du modèle depuis Google Drive...")
    try:
        import gdown
        # Votre lien Google Drive
        url = "https://drive.google.com/drive/folders/1WpDMYwGEbsxzD5BjUqyA2ajbSHjjxz63"
        gdown.download(url, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Taille en MB
            print(f" Modèle téléchargé avec succès! Taille: {file_size:.2f} MB")
            return True
        else:
            print(" Échec du téléchargement - fichier non créé")
            return False
            
    except Exception as e:
        print(f" Erreur lors du téléchargement: {e}")
        return False

# Charger le modèle (avec auto-download)
model = None
IMG_SIZE = 128

if TENSORFLOW_AVAILABLE:
    try:
        if download_model():
            MODEL_PATH = "best_model.h5"
            model = load_model(MODEL_PATH)
            print(" Modèle chargé avec succès")
        else:
            print(" Impossible de charger le modèle")
            model = None
    except Exception as e:
        print(f" Erreur chargement modèle: {e}")
        model = None
else:
    print(" TensorFlow non disponible - mode démo seulement")
    model = None

def predict_image(image_path):
    if model is None:
        return "Erreur: Modèle non chargé", 0.0
        
    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array, verbose=0)[0]
        classes = ['Parasitized', 'Uninfected']
        label = classes[np.argmax(pred)]
        confidence = float(np.max(pred))
        return label, confidence
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return "Erreur", 0.0

def predict_folder(folder_path):
    results = []
    total = 0
    infected = 0
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not os.path.isfile(img_path):
            continue
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            label, conf = predict_image(img_path)
            if label != "Erreur":
                results.append((img_name, label, conf))
                total += 1
                if label == "Parasitized":
                    infected += 1
        except Exception as e:
            print(f"Erreur analyse {img_name}: {e}")
            continue
            
    infection_rate = (infected / total) * 100 if total > 0 else 0
    return results, infection_rate

def secure_filename(filename):
    """Sécuriser le nom de fichier"""
    filename = os.path.basename(filename)
    keepchars = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()

@app.route('/')
def index():
    if model is None:
        flash(" Le service est temporairement indisponible. Le modèle est en cours de chargement.", "warning")
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if model is None:
        flash('Service temporairement indisponible. Le modèle est en cours de chargement.', 'error')
        return redirect(url_for('index'))
        
    if 'image' not in request.files:
        flash('Aucune image sélectionnée.', 'error')
        return redirect(url_for('index'))
        
    file = request.files['image']
    if file.filename == '':
        flash('Aucune image sélectionnée.', 'error')
        return redirect(url_for('index'))
    
    try:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        label, conf = predict_image(path)
        return render_template('result_single.html', image_path=path, label=label, confidence=conf)
    except Exception as e:
        print(f"Erreur traitement image: {e}")
        flash('Erreur lors du traitement de l\'image.', 'error')
        return redirect(url_for('index'))

@app.route('/predict_folder', methods=['POST'])
def predict_folder_route():
    if model is None:
        flash('Service temporairement indisponible. Le modèle est en cours de chargement.', 'error')
        return redirect(url_for('index'))
        
    if 'folder' not in request.files:
        flash('Aucun dossier sélectionné.', 'error')
        return redirect(url_for('index'))
    
    folder_files = request.files.getlist('folder')
    if not folder_files or folder_files[0].filename == '':
        flash('Aucun dossier sélectionné.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Utiliser un dossier fixe pour simplifier
        folder_path = os.path.join(UPLOAD_FOLDER, 'patient_folder')
        
        # Vider et recréer le dossier
        shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path, exist_ok=True)
        
        # Sauvegarder les fichiers
        saved_count = 0
        for file in folder_files:
            if file.filename == '':
                continue
            filename = secure_filename(os.path.basename(file.filename))
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file.save(os.path.join(folder_path, filename))
                saved_count += 1
        
        if saved_count == 0:
            flash('Aucune image valide trouvée dans le dossier.', 'error')
            return redirect(url_for('index'))
            
        results, infection_rate = predict_folder(folder_path)
        return render_template('result_folder.html', results=results, infection_rate=infection_rate)
        
    except Exception as e:
        print(f"Erreur traitement dossier: {e}")
        flash('Erreur lors du traitement du dossier.', 'error')
        return redirect(url_for('index'))

# Gestion des erreurs
@app.errorhandler(413)
def too_large(e):
    flash('Fichier trop volumineux. Maximum 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('Erreur interne du serveur.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)