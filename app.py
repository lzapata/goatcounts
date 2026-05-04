import streamlit as st
import requests
import base64
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# Configuración visual
st.set_page_config(page_title="Goat Counter AI", page_icon="🐐")
st.markdown("""
    <style>
    .main { background-color: #0f1a0f; color: #e2f5e2; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #4ade80; color: #0f1a0f; font-weight: bold; }
    </style>
    """, unsafe_allow_ Harris=True)

st.title("🐐 Goat Detector AI")

# Sidebar para configuración
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Roboflow API Key", type="password")
model_id = st.sidebar.selectbox("Modelo", [
    "brookside-research/goat-looker/1",
    "justin-burger/goats-hqnax/1"
])
conf_threshold = st.sidebar.slider("Confianza", 0.1, 1.0, 0.4)

# Historial en memoria
if 'history' not in st.session_state:
    st.session_state.history = []

# Captura de cámara
img_file = st.camera_input("Toma una foto de las cabras")

if img_file and api_key:
    # Convertir imagen para Roboflow
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Encode a Base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Llamada a la API de Roboflow (Desde el servidor, sin bloqueos de navegador)
    url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
    
    with st.spinner('Analizando...'):
        response = requests.post(url, data=img_base64, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        result = response.json()

    predictions = result.get('predictions', [])

    if predictions:
        for p in predictions:
            # Coordenadas
            x = int(p['x'] - p['width'] / 2)
            y = int(p['y'] - p['height'] / 2)
            w = int(p['width'])
            h = int(p['height'])
            conf = int(p['confidence'] * 100)

            # Dibujar en la imagen
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (74, 222, 128), 8)
            label = f"GOAT {conf}%"
            cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (74, 222, 128), 4)

        st.image(img_array, caption=f"Detectadas: {len(predictions)} cabras")

        # Botón de guardar
        if st.button("💾 GUARDAR REGISTRO"):
            avg_conf = int(np.mean([p['confidence'] for p in predictions]) * 100)
            nuevo_registro = {
                "fecha": datetime.now().strftime("%H:%M:%S"),
                "animal": "GOAT",
                "cantidad": len(predictions),
                "efectividad": f"{avg_conf}%"
            }
            st.session_state.history.append(nuevo_registro)
            st.success("¡Registro guardado!")

    else:
        st.warning("No se detectaron cabras. Intenta con más luz o bajando la confianza.")

# Mostrar historial
if st.session_state.history:
    st.write("### 📋 Historial de hoy")
    for reg in reversed(st.session_state.history):
        st.info(f"🕒 {reg['fecha']} | {reg['animal']} x{reg['cantidad']} | Efectividad: {reg['efectividad']}")