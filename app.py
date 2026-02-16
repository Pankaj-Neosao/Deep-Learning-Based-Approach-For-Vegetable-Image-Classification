import os
import json
import io
import logging
from PIL import Image
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf   # ✅ USE TENSORFLOW
import gdown

# =======================
# Logging
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# Model Download Setup
# =======================
DRIVE_FILE_ID = "1rEKE-68SDdAOu8_GmycTgf3nmKJ_GMeA"
MODEL_FILENAME = "vegetable_model.tflite"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading TFLite model from Google Drive...")

    gdown.download(
        DOWNLOAD_URL,
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

    # Prevent corrupted HTML download
    if os.path.getsize(MODEL_PATH) < 1000000:
        raise ValueError("Downloaded file is too small. Not a valid TFLite model.")

    logger.info("Model downloaded successfully.")

# =======================
# Vegetable Classifier
# =======================
class VegetableClassifier:
    def __init__(self, model_path=MODEL_PATH, class_indices_filename="class_indices.json"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")

        class_indices_path = os.path.join(BASE_DIR, class_indices_filename)
        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(f"Class indices file not found at: {class_indices_path}")

        logger.info("Loading TFLite model...")

        # ✅ TensorFlow Lite Interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info("TFLite model loaded successfully.")

        with open(class_indices_path, "r") as f:
            self.class_indices = json.load(f)

        self.class_names = {v: k for k, v in self.class_indices.items()}
        self.img_height = 224
        self.img_width = 224

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_bytes):
        processed_image = self.preprocess_image(image_bytes)

        self.interpreter.set_tensor(self.input_details[0]["index"], processed_image)
        self.interpreter.invoke()

        predictions = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

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
# Flask App
# =======================
app = Flask(__name__)
classifier = VegetableClassifier()

@app.route("/")
def index():
    return render_template(
        "index.html",
        status="ok",
        message="TFLite Model loaded successfully 🚀",
    )

@app.route("/predict", methods=["POST"])
def predict():
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
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
