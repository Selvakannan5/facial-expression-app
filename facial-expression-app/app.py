import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.inference import predict_expression

st.set_page_config(layout="centered")
st.title("Facial Expression Recognition (CNN Model)")

uploaded_image = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)
    prediction = predict_expression(image_np)
    st.image(image, caption=f"Predicted Expression: {prediction}", use_column_width=True)