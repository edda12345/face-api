from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json

app = Flask(name)

# Load model and labels
model = tf.keras.models.load_model("best_face_model.h5")
with open("label_list.json", "r") as f:
    labels = json.load(f)

def preprocess_image(base64_str):
    image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    img_array = preprocess_image(data["image"])
    predictions = model.predict(img_array)[0]
    idx = int(np.argmax(predictions))
    confidence = float(predictions[idx])

    if confidence >= 0.5:
        return jsonify({"label": labels[idx], "confidence": confidence})
    else:
        return jsonify({"label": "Unknown", "confidence": confidence})

if name == "main":
    app.run(host="0.0.0.0", port=10000)
