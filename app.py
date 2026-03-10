from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # السماح للموقع بالتواصل مع الـ backend

# تحميل الموديل مرة واحدة عند التشغيل
MODEL_PATH = "brain_tumor_model.h5"

try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_loaded = False

# ======================================================
# عدّل هذه الإعدادات حسب موديلك
# ======================================================
IMG_SIZE = (224, 224)        # حجم الصورة المطلوب (غيّره لو الموديل بتاعك مختلف)
CLASS_NAMES = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
# لو الموديل binary (ورم / لا ورم) استخدم:
# CLASS_NAMES = ["No Tumor", "Tumor Detected"]
# ======================================================

def preprocess_image(image_bytes):
    """تجهيز الصورة قبل إدخالها للموديل"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # إضافة batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        # التنبؤ
        predictions = model.predict(img_array)[0]

        # لو الموديل binary (neuron واحد بـ sigmoid)
        if len(predictions) == 1:
            confidence = float(predictions[0])
            if confidence > 0.5:
                result = {"class": "Tumor Detected", "confidence": confidence, "all_predictions": {}}
            else:
                result = {"class": "No Tumor", "confidence": 1 - confidence, "all_predictions": {}}
        else:
            # Multi-class
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class {predicted_idx}"
            confidence = float(predictions[predicted_idx])

            all_preds = {}
            for i, prob in enumerate(predictions):
                label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
                all_preds[label] = round(float(prob) * 100, 2)

            result = {
                "class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "all_predictions": all_preds
            }

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model_loaded,
        "classes": CLASS_NAMES
    })


if __name__ == "__main__":
    print("🧠 Brain Tumor Detection API")
    print(f"📁 Model path: {MODEL_PATH}")
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
