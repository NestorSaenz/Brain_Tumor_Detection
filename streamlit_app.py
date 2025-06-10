import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import gdown
import requests

MODEL_URL = "https://drive.google.com/file/d/15TvDCVIne0eect8Rc9IcLyehdao48fvl/view?usp=sharing" 
MODEL_PATH = "best_resnet_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

download_model()

# Configuración de la página
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.markdown(
    """
    <style>
    .main-title {font-size:3em; font-weight:bold; color:#4F8BF9;}
    .subtitle {font-size:1.3em; color:#333;}
    .footer {font-size:0.9em; color:#888; margin-top: 2em;}
    .stButton>button {background-color: #4F8BF9; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube una imagen de resonancia magnética cerebral (MRI) y el modelo clasificará si hay presencia de tumor.</div>', unsafe_allow_html=True)

# Cargar el modelo (usa cache para eficiencia)
@st.cache_resource
def load_brain_model():
    return load_model(MODEL_PATH)

model = load_brain_model()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=120)
st.sidebar.title("Opciones")
st.sidebar.markdown(
    """
    - Formato recomendado: JPG, PNG
    - Tamaño de entrada: 224x224 px
    - Modelo: ResNet50 Fine-Tuned
    """
)

# Subida de imagen
uploaded_file = st.file_uploader("Selecciona una imagen de MRI cerebral", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    st.markdown("---")
    st.write("Procesando imagen...")

    # Preprocesar y predecir
    img_batch = preprocess_image(image)
    prediction = model.predict(img_batch)
    prob = float(prediction[0][0])
    label = "Tumor" if prob > 0.5 else "Healthy"
    color = "#e74c3c" if label == "Tumor" else "#27ae60"
    st.markdown(
        f'<h2 style="color:{color};text-align:center;">Resultado: {label}</h2>',
        unsafe_allow_html=True
    )
    st.progress(int(prob*100) if label == "Tumor" else int((1-prob)*100))
    st.markdown(
        f"<div style='text-align:center;'>Probabilidad de tumor: <b>{prob*100:.2f}%</b></div>" if label == "Tumor"
        else f"<div style='text-align:center;'>Probabilidad de estar sano: <b>{(1-prob)*100:.2f}%</b></div>",
        unsafe_allow_html=True
    )
else:
    st.info("Por favor, sube una imagen de MRI cerebral para analizar.")

# Footer
st.markdown(
    '<div class="footer">Desarrollado por tu equipo de IA | Proyecto Bootcamp 2025</div>',
    unsafe_allow_html=True
)