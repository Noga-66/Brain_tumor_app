import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model ──
import keras
keras.config.enable_unsafe_deserialization()
import tensorflow as tf

model = tf.keras.models.load_model("brain_tumor_model.h5", safe_mode=False)
IMG_SIZE = (256, 256)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr, verbose=0)

        # binary sigmoid output — shape could be (1,1,1,1) or (1,1) depending on model
        score = float(np.squeeze(pred))
        tumor_conf   = round(score * 100, 2)
        no_tumor_conf = round((1 - score) * 100, 2)

        if score >= 0.5:
            label = "Tumor Detected"
            confidence = tumor_conf
            has_tumor = True
        else:
            label = "No Tumor"
            confidence = no_tumor_conf
            has_tumor = False

        return JSONResponse({
            "label": label,
            "confidence": confidence,
            "has_tumor": has_tumor,
            "tumor_prob": tumor_conf,
            "no_tumor_prob": no_tumor_conf
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
