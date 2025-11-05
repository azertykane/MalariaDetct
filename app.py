import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malaria_secret_key_prod")

IMG_SIZE = 128
model = None


try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow non disponible : {e}")
    TENSORFLOW_AVAILABLE = False

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
        url = "https://drive.google.com/uc?id=1WpDMYwGEbsxzD5BjUqyA2ajbSHjjxz63"  #  lien direct vers le .h5
        gdown.download(url, MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        print(f"Erreur téléchargement modèle : {e}")
        return False

if TENSORFLOW_AVAILABLE:
    try:
        if download_model():
            model = load_model("best_model.h5")
            print(" Modèle chargé avec succès.")
        else:
            print(" Impossible de charger le modèle.")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
else:
    print("TensorFlow non disponible — mode démo actif.")


def predict_image_stream(file_stream):
    """Analyse une image directement depuis la mémoire (sans sauvegarde)"""
    if model is None:
        return "Erreur: modèle non chargé", 0.0

    try:
        img = Image.open(BytesIO(file_stream.read())).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0]
        classes = ['Parasitized', 'Uninfected']
        label = classes[np.argmax(pred)]
        confidence = float(np.max(pred))
        return label, confidence
    except Exception as e:
        print(f"Erreur analyse mémoire : {e}")
        return "Erreur", 0.0

def predict_folder_memory(files):
    """Analyse plusieurs images sans les enregistrer"""
    results = []
    infected = 0
    total = 0

    for file in files:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            label, conf = predict_image_stream(file)
            if label != "Erreur":
                results.append((file.filename, label, conf))
                total += 1
                if label == "Parasitized":
                    infected += 1
        except Exception as e:
            print(f"Erreur sur {file.filename}: {e}")
            continue

    infection_rate = (infected / total) * 100 if total > 0 else 0
    return results, infection_rate


@app.route('/')
def index():
    if model is None:
        flash(" Le modèle est en cours de chargement, veuillez patienter.", "warning")
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if 'image' not in request.files:
        flash("Aucune image sélectionnée.", "error")
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash("Aucune image sélectionnée.", "error")
        return redirect(url_for('index'))

    try:
        label, conf = predict_image_stream(file)
        return render_template('result_single.html', label=label, confidence=conf)
    except Exception as e:
        print(f"Erreur traitement image : {e}")
        flash("Erreur lors du traitement de l'image.", "error")
        return redirect(url_for('index'))

@app.route('/predict_folder', methods=['POST'])
def predict_folder_route():
    files = request.files.getlist('folder')
    if not files or files[0].filename == '':
        flash("Aucun dossier sélectionné.", "error")
        return redirect(url_for('index'))

    try:
        results, infection_rate = predict_folder_memory(files)
        return render_template('result_folder.html', results=results, infection_rate=infection_rate, patient_folder="memory")
    except Exception as e:
        print(f"Erreur dossier mémoire : {e}")
        flash("Erreur lors du traitement du dossier.", "error")
        return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('Erreur interne du serveur.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
