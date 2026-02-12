# =======================
# app.py
# =======================

import os

# (Optional) Disable oneDNN logs for clean output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging
import json
import io
import os
from flask import Flask, request, jsonify
from flask import render_template


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
        model_path="vegetable_classification_model.h5",
        class_indices_path="class_indices.json",
    ):
        # Always load files relative to this script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.normpath(os.path.join(BASE_DIR, model_path))
        class_indices_path = os.path.normpath(
            os.path.join(BASE_DIR, class_indices_path)
        )

        logger.info(f"Looking for model at: {model_path}")
        logger.info(f"Looking for class indices at: {class_indices_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(
                f"Class indices file not found at: {class_indices_path}"
            )

        # Load model
        logger.info("Loading model...")
        self.model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

        # Load class indices
        with open(class_indices_path, "r") as f:
            self.class_indices = json.load(f)

        # Reverse mapping: index → class name
        self.class_names = {v: k for k, v in self.class_indices.items()}

        # Image parameters (must match training)
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

        # Top-3 predictions
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
    """Used when the real model cannot be loaded."""
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
    """Renders the main HTML page and passes the model status."""
    if classifier:
        if using_mock:
            status = "warning"
            message = (
                f"Warning: Model not found. Running in MOCK mode. (Error: {init_error})"
            )
        else:
            status = "ok"
            message = "Model loaded. Ready to receive predictions at /predict."
    else:
        status = "error"
        message = (
            f"Model failed to load. Error: {init_error}. Check server logs for paths."
        )
    return render_template("index.html", status=status, message=message)


@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None:
        return jsonify({"error": "Model is not available on the server."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    result = classifier.predict(image_bytes)

    return jsonify(result)


# =======================
# Run Server
# =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
