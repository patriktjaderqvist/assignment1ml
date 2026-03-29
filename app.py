import streamlit as st
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import requests

st.title("Imagenette Image Classifier")
st.write("Upload an image to classify it into one of 10 categories: tench, english springer, cassette player, chain saw, church, french horn, garbage truck, gas pump, golf ball, parachute")

learn = load_learner(Path("imagenette_classifier.pkl"))

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    pred, idx, probs = learn.predict(img)
    st.success(f"Prediction: **{pred}**")
    st.write(f"Confidence: {float(probs[idx]):.2%}")
    st.bar_chart({learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))})
