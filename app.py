import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_disease_final.keras")

# Load class names
@st.cache_data
def load_class_names():
    with open("models/class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if st.button("🔍 Detect Disease"):

        predictions = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100

        predicted_class = class_names[predicted_index]

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")