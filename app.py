from flask import Flask, request, jsonify
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite

app = Flask(name)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="best_face_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
labels = ["Anastasia", "Edda", "Esther", "Kimberly", "Roshini", "Amylea"]

def preprocess_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        input_tensor = preprocess_image(data["image"])
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        prediction_index = int(np.argmax(output_data))
        confidence = float(output_data[prediction_index])

        return jsonify({
            "label": labels[prediction_index],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Face Recognition API is Running!"

if name == "main":
    app.run(debug=True)
