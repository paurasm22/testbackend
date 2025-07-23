from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load your model
model = load_model("MobileNetv2.h5")
class_names = ["broken_benches", "garbage", "potholes", "streetlight"]  # customize as per your classes

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # MobileNetV2 default size
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]
    img = preprocess_image(file.read())

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction[0])]

    return jsonify({"class": predicted_class, "confidence": float(np.max(prediction))})

if __name__ == "__main__":
    app.run(debug=True)
