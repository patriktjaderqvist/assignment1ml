import streamlit as st
from fastai.vision.all import load_learner, PILImage
from pathlib import Path

st.set_page_config(page_title="Imagenette Classifier", page_icon="🔍", layout="centered")

st.markdown("## Image Classifier — Imagenette")
st.markdown("Trained on 10 classes using transfer learning (ResNet34, ~96.8% accuracy)")
st.markdown("---")

learn = load_learner(Path("imagenette_classifier.pkl"))

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Classes:**")
    for c in learn.dls.vocab:
        st.markdown(f"- {c}")

with col2:
    uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        st.image(uploaded_file, use_column_width=True)
        pred, idx, probs = learn.predict(img)
        top3 = sorted(zip(learn.dls.vocab, map(float, probs)), key=lambda x: -x[1])[:3]
        st.markdown(f"### → {pred} ({float(probs[idx]):.1%})")
        st.markdown("**Top 3:**")
        for label, prob in top3:
            st.progress(prob, text=f"{label}: {prob:.1%}")
