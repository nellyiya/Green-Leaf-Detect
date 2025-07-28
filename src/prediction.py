import sys
import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (128, 128)
MODEL_PATH = "models/plant_disease_model.keras"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0,1]
    img = tf.expand_dims(img, axis=0)  # add batch dimension
    return img

def predict(image_path, model, class_names):
    img = preprocess_image(image_path)
    preds = model.predict(img)[0]
    top_3_indices = preds.argsort()[-3:][::-1]
    print("Top 3 Predictions:")
    for i in top_3_indices:
        print(f"{class_names[i]}: {preds[i]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prediction.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()

    # Assuming you saved class names in a file or define here
    # If not, you can hardcode class names or load from training metadata
    # Example (adjust to your dataset classes):
    class_names = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___Healthy",
        "Potato___Early_blight",
        "Potato___Late_blight"
    ]

    predict(image_path, model, class_names)
