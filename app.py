import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")
BASE = Path(__file__).parent
CLASS_NAMES = json.load(open(BASE / "class_names.json"))
IMG_SIZE = tuple(json.load(open(BASE / "model_info.json"))["img_size"])


@st.cache_resource
def load_model():
    keras_path, tflite_path = BASE / "skin_cancer_resnet50.keras", BASE / "skin_cancer_resnet50.tflite"
    if keras_path.exists():
        m = tf.keras.models.load_model(keras_path)
        return lambda x: m.predict(x, verbose=0)[0]
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def predict(x):
        interp.set_tensor(inp["index"], x.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out["index"])[0]
    return predict


predict = load_model()

st.title("🩺 Skin Cancer Detection")
st.caption("ResNet50 transfer-learning model. Educational demo only - NOT a medical diagnosis.")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    x = np.expand_dims(np.asarray(img.resize(IMG_SIZE), dtype=np.float32), 0)
    probs = predict(preprocess_input(x))
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.metric("Confidence", f"{probs[top]*100:.1f}%")
    st.bar_chart({c: float(p) for c, p in zip(CLASS_NAMES, probs)})
    st.warning("Please consult a qualified dermatologist for any real medical concern.")
