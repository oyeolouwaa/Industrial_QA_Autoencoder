import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from model import ConvAutoencoder

st.title("Contrôle Qualité IA — Détection d'Anomalies")

@st.cache_resource
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('autoencoder_anomaly.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

threshold = st.slider("Seuil de tolérance du bruit", min_value=0.0, max_value=0.2, value=0.08, step=0.01)

uploaded_file = st.file_uploader("Charger l'image d'une pièce à contrôler...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Pièce testée", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(input_tensor)

    orig_np = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    recon_np = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
    
    diff = np.mean(np.abs(orig_np - recon_np), axis=2)
    
    anomaly_score = float(np.sum(diff > threshold) / diff.size)
    
    if anomaly_score > 0.01:
        st.error(f"⚠️ **Anomalie détectée !** (Sévérité : {anomaly_score:.4f})")
    else:
        st.success(f"✅ **Pièce conforme** (Sévérité : {anomaly_score:.4f})")

    diff_filtered = diff.copy()
    diff_filtered[diff_filtered < threshold] = 0
    
    diff_norm = cv2.normalize(diff_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    orig_cv = np.array(image)
    overlay = cv2.addWeighted(orig_cv, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Carte de chaleur des anomalies détectées", use_container_width=True)