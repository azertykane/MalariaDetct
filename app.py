import os
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# Initialisation de Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malaria_secret_key_prod")

IMG_SIZE = 128
model = None

# ================================
#       BASE DE DONNÉES
# ================================
DB_PATH = "users.db"

def init_db():
    """Crée la base SQLite pour les utilisateurs"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

init_db()

# ================================
#     CHARGEMENT DU MODÈLE
# ================================
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    import gdown
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow non disponible : {e}")
    TENSORFLOW_AVAILABLE = False


def download_model():
    """Télécharge le modèle depuis Google Drive si absent"""
    MODEL_PATH = "best_model.h5"
    if os.path.exists(MODEL_PATH):
        print("✅ Modèle déjà présent")
        return True
    try:
        url = "https://drive.google.com/uc?id=1Dw8LOmHC3qaQPpLkhR79eTr_qWBIui9l"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        print(f"Erreur téléchargement modèle : {e}")
        return False


if TENSORFLOW_AVAILABLE:
    try:
        if download_model():
            model = load_model("best_model.h5")
            print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")

# ================================
#         PREDICTION
# ================================
def predict_image_stream(file_stream):
    """Analyse une image directement depuis la mémoire"""
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


# ================================
#         ROUTES AUTH
# ================================
@app.before_request
def before_request():
    g.user = session.get("user")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash("Veuillez remplir tous les champs.", "warning")
            return redirect(url_for('register'))

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("✅ Compte créé avec succès, vous pouvez vous connecter.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("⚠️ Nom d'utilisateur déjà pris.", "error")
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            flash(f"Bienvenue, {username} 👋", "success")
            return redirect(url_for('index'))
        else:
            flash("Identifiants invalides.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Déconnexion réussie.", "info")
    return redirect(url_for('home'))


# ================================
#     ROUTES DÉTECTION
# ================================
@app.route('/index')
def index():
    if not g.user:
        flash("Veuillez vous connecter pour accéder à l'analyse.", "warning")
        return redirect(url_for('login'))
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

    import base64
    image_bytes = file.read()
    label, conf = predict_image_stream(BytesIO(image_bytes))
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{img_base64}"

    return render_template('result_single.html', label=label, confidence=conf, image_data=image_data)
@app.route('/predict_folder', methods=['POST'])
def predict_folder():
    try:
        if 'folder' not in request.files:
            flash("Aucun dossier n'a été sélectionné", "error")
            return redirect(url_for('index'))

        files = request.files.getlist('folder')
        if len(files) == 0:
            flash("Le dossier est vide", "error")
            return redirect(url_for('index'))

        patient_folder = os.path.join('static', 'uploads', 'memory', 'test2')
        os.makedirs(patient_folder, exist_ok=True)

        for f in os.listdir(patient_folder):
            os.remove(os.path.join(patient_folder, f))

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(patient_folder, filename)
            file.save(file_path)

        print(f"[INFO] Dossier enregistré : {patient_folder}")

        model_path = "best_model.h5"
        model = load_model(model_path)

        results = []
        for filename in os.listdir(patient_folder):
            file_path = os.path.join(patient_folder, filename)
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            pred_label = "Parasitized" if prediction[0][0] > 0.5 else "Uninfected"

            prob = float(prediction[0][0])
            if pred_label == "Parasitized":
               conf = prob * 100
            else:
               conf = (1 - prob) * 100
            results.append({
                "filename": filename,
                "label": pred_label,
                "probability": conf / 100
            })

        parasitized = [r for r in results if r['label'] == "Parasitized"]
        uninfected = [r for r in results if r['label'] == "Uninfected"]
        total = len(results)
        infected = len(parasitized)
        infection_rate = (infected / total) * 100 if total > 0 else 0

        return render_template("result_folder.html",
                               parasitized=parasitized,
                               uninfected=uninfected,
                               infection_rate=infection_rate,
                               results=[(r['filename'], r['label'], r['probability']) for r in results],
                               patient_folder="memory/test2")

    except Exception as e:
        print("Erreur traitement dossier :", e)
        flash("Erreur lors du traitement du dossier", "error")
        return redirect(url_for('index'))



# ================================
#        ERREUR SERVER
# ================================
@app.errorhandler(500)
def internal_error(error):
    flash('Erreur interne du serveur.', 'error')
    return redirect(url_for('index'))


# ================================
#        RUN SERVEUR
# ================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
