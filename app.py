import os
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image

# ================================
#       FLASK INIT
# ================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malaria_secret_key_prod")

IMG_SIZE = 128
model = None
DB_PATH = "users.db"


# ================================
#           DATABASE
# ================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ================================
#      MODEL LOADING
# ================================
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    import gdown
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow non disponible")
    TENSORFLOW_AVAILABLE = False


def download_model():
    """Télécharge le modèle si absent"""
    MODEL_PATH = "best_model.h5"
    if os.path.exists(MODEL_PATH):
        print("📦 Modèle déjà présent.")
        return True

    try:
        url = "https://drive.google.com/uc?id=1Dw8LOmHC3qaQPpLkhR79eTr_qWBIui9l"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Erreur téléchargement : {e}")
        return False


if TENSORFLOW_AVAILABLE:
    try:
        if download_model():
            model = load_model("best_model.h5")
            print("✅ Modèle chargé avec succès.")
        else:
            print("❌ Impossible de charger le modèle.")
    except Exception as e:
        print(f"⚠️ Erreur chargement modèle : {e}")


# ================================
#       PREDICTION LOGIC
# ================================
def predict_image_stream(file_stream):
    """Analyse une image en mémoire"""
    if model is None:
        return "Erreur", 0.0

    try:
        img = Image.open(BytesIO(file_stream.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)

        pred = model.predict(img_array, verbose=0)[0]
        label = "Parasitized" if pred[0] > 0.5 else "Uninfected"
        confidence = float(pred[0] if pred[0] > 0.5 else 1 - pred[0])
        return label, confidence

    except Exception as e:
        print("Erreur analyse :", e)
        return "Erreur", 0.0


# ================================
#       AUTH MIDDLEWARE
# ================================
@app.before_request
def before_request():
    g.user = session.get("user")


# ================================
#       AUTH ROUTES
# ================================
@app.route('/')
def home():
    return render_template("home.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash("Veuillez remplir tous les champs.", "warning")
            return redirect(url_for("register"))

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("✅ Compte créé avec succès.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Nom d'utilisateur déjà utilisé.", "error")

    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            flash("Connexion réussie.", "success")
            return redirect(url_for("index"))
        else:
            flash("Identifiants incorrects.", "error")

    return render_template("login.html")


@app.route('/logout')
def logout():
    session.pop("user", None)
    flash("Déconnecté.", "info")
    return redirect(url_for("home"))


# ================================
#       DETECTION ROUTES
# ================================
@app.route('/index')
def index():
    if not g.user:
        flash("Veuillez vous connecter.", "warning")
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route('/predict_single', methods=['POST'])
def predict_single():
    if 'image' not in request.files:
        flash("Aucune image envoyée.", "error")
        return redirect(url_for("index"))

    file = request.files['image']
    if file.filename == '':
        flash("Fichier vide.", "error")
        return redirect(url_for("index"))

    image_bytes = file.read()
    label, conf = predict_image_stream(BytesIO(image_bytes))

    import base64
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data = "data:image/jpeg;base64," + img_base64

    return render_template("result_single.html", label=label, confidence=conf, image_data=image_data)


@app.route('/predict_folder', methods=['POST'])
def predict_folder():
    try:
        files = request.files.getlist('folder')

        if not files:
            flash("Aucun fichier.", "error")
            return redirect(url_for('index'))

        folder_path = "static/uploads/memory/test2"
        os.makedirs(folder_path, exist_ok=True)

        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))

        # Save files
        for file in files:
            file.save(os.path.join(folder_path, secure_filename(file.filename)))

        # Prediction
        results = []
        for filename in os.listdir(folder_path):
            img = Image.open(os.path.join(folder_path, filename)).resize((128, 128))
            arr = np.expand_dims(np.array(img) / 255.0, 0)
            pred = model.predict(arr, verbose=0)[0][0]
            label = "Parasitized" if pred > 0.5 else "Uninfected"
            conf = float(pred if pred > 0.5 else 1 - pred)
            results.append({"filename": filename, "label": label, "probability": conf})

        parasitized = [r for r in results if r["label"] == "Parasitized"]
        uninfected = [r for r in results if r["label"] == "Uninfected"]
        infection_rate = (len(parasitized) / len(results)) * 100

        return render_template(
            "result_folder.html",
            parasitized=parasitized,
            uninfected=uninfected,
            infection_rate=infection_rate,
            patient_folder="memory/test2"
        )

    except Exception as e:
        print("Erreur dossier :", e)
        flash("Erreur analyse dossier.", "error")
        return redirect(url_for("index"))


# ================================
#       ERROR HANDLER
# ================================
@app.errorhandler(500)
def internal_error(error):
    flash("Erreur serveur.", "error")
    return redirect(url_for("index"))


# ================================
#       RUN SERVER
# ================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
