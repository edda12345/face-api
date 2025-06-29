from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(name)

# Load the model
model = tf.keras.models.load_model("best_face_model.h5")

# Your class labels
labels = ["Anastasia", "Edda", "Esther", "Kimberly", "Roshini", "Amylea"]

def preprocess_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        input_tensor = preprocess_image(data["image"])
        prediction = model.predict(input_tensor)[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])
        return jsonify({
            "label": labels[predicted_index],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Face recognition API is live!"

if name == "main":
    app.run(debug=True)
