# =======================
# app.py
# =======================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging
import json
import io
import gdown
from flask import Flask, request, jsonify, render_template

# =======================
# Logging Configuration
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# Vegetable Classifier
# =======================
class VegetableClassifier:
    def __init__(
        self,
        model_filename="vegetable_classification_model.h5",
        class_indices_filename="class_indices.json",
    ):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, model_filename)
        class_indices_path = os.path.join(BASE_DIR, class_indices_filename)

        # -------- DOWNLOAD MODEL IF NOT EXISTS --------
        if not os.path.exists(model_path):
            logger.info("Model not found locally. Downloading from Google Drive...")

            # 🔴 Replace only if you change Drive file
            file_id = "1qwZhdagJq74b2DZMJX6Afp9jAImlFZIk"
            url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(url, model_path, quiet=False)

        # -------- FINAL CHECK --------
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(
                f"Class indices file not found at: {class_indices_path}"
            )

        logger.info("Loading model...")
        self.model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

        with open(class_indices_path, "r") as f:
            self.class_indices = json.load(f)

        self.class_names = {v: k for k, v in self.class_indices.items()}

        self.img_height = 224
        self.img_width = 224

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_bytes):
        processed_image = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed_image, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])

        top_indices = predictions.argsort()[-3:][::-1]
        top_predictions = [
            {
                "vegetable": self.class_names[int(idx)],
                "confidence": float(predictions[idx]),
            }
            for idx in top_indices
        ]

        return {
            "predicted_vegetable": self.class_names[int(predicted_idx)],
            "confidence": confidence,
            "top_predictions": top_predictions,
        }


# =======================
# Mock Classifier (Fallback)
# =======================
class MockVegetableClassifier:
    def predict(self, image_bytes):
        return {
            "predicted_vegetable": "Mock Tomato (Test)",
            "confidence": 0.99,
            "top_predictions": [
                {"vegetable": "Mock Tomato (Test)", "confidence": 0.99},
                {"vegetable": "Mock Potato", "confidence": 0.005},
                {"vegetable": "Mock Carrot", "confidence": 0.005},
            ],
        }


# =======================
# Flask App
# =======================
app = Flask(__name__)

classifier = None
init_error = None
using_mock = False

try:
    classifier = VegetableClassifier()
    logger.info("VegetableClassifier initialized.")
except Exception as e:
    logger.error(f"Error initializing classifier: {e}")
    init_error = str(e)
    logger.warning("Falling back to MockVegetableClassifier.")
    classifier = MockVegetableClassifier()
    using_mock = True


@app.route("/")
def index():
    if using_mock:
        status = "warning"
        message = (
            f"Warning: Model not loaded. Running in MOCK mode. (Error: {init_error})"
        )
    else:
        status = "ok"
        message = "Model loaded successfully. Ready for predictions at /predict."

    return render_template("index.html", status=status, message=message)


@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None:
        return jsonify({"error": "Model not available."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    result = classifier.predict(image_bytes)

    return jsonify(result)


# =======================
# Run Server (Local Only)
# =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
