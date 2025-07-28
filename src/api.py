from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import shutil
import os
import asyncio

# Import your build_and_train retraining function
from .model import build_and_train


app = FastAPI()

MODEL_PATH = "models/plant_disease_model.keras"
IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

# Your classes - update if needed
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight"
]

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Use JPEG or PNG.")
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)[0]
    top3_indices = preds.argsort()[-3:][::-1]

    results = []
    for i in top3_indices:
        results.append({"class": class_names[i], "confidence": float(preds[i])})

    return {"predictions": results}

@app.post("/upload/")
async def upload_data(files: list[UploadFile] = File(...)):
    """
    Endpoint to upload bulk images for retraining.
    Saves uploaded files to data/train directory under appropriate class folders.
    """
    # Example: For demo, we expect the client to send the class name in the filename as prefix, e.g. Pepper__bell___Healthy_1.jpg
    for file in files:
        filename = file.filename
        # Extract class name prefix from filename (assumes format: ClassName_filename.jpg)
        try:
            class_name = filename.split('_')[0]
        except IndexError:
            class_name = "Unknown"

        class_dir = os.path.join("data", "train", class_name)
        os.makedirs(class_dir, exist_ok=True)
        file_path = os.path.join(class_dir, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return {"message": f"Uploaded {len(files)} files successfully."}

@app.post("/retrain/")
async def retrain():
    """
    Trigger model retraining.
    Runs the build_and_train function asynchronously.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_and_train)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)  # Reload updated model

    return {"message": "Model retrained and updated successfully."}
