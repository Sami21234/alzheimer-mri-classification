
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config

st.set_page_config(
    page_title="Alzheimer MRI Detection",
    page_icon=":brain:",
    layout="wide"
)

# Custom CSS

st.markdown("""
<style>
.main {
    background-color: #f8ffff;
}
.card {
    background-color: #ffffff;
    padding: 0.2rem;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0, 180, 180, 0.15);
    margin-bottom: 1.2rem;
}
.title {
    color: #00cfd1;
    font-size: 38px;
    font-weight: 700;
}
.subtitle {
    color: #00cfd1;
    font-size: 20px;
    font-weight: 600;
}
.result {
    font-size: 28px;
    font-weight: 700;
    color: #00cfd1;
}
</style>
""", unsafe_allow_html=True)

# Title

st.markdown('<div class="title">🧠 Alzheimer’s Disease Detection (MRI)</div>', unsafe_allow_html=True)
st.write("Early-stage classification using Deep Learning (ResNet50)")

# Load model

@st.cache_resource    # Cache the model loading to improve performance.
def load_model():     # function to load the model.
    return tf.keras.models.load_model("models/best_resnet_model.h5")

model = load_model()

# Class labels for the model(must be same as in the training phase)
class_names = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very Mild Dementia"
]

# Layout

left_col, right_col = st.columns([1, 2])

# LEFT COLUMN — Upload

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">📤 Upload MRI Scan</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(    # File uploader for images.
        "Choose MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    st.info("Supported formats: JPG, PNG\n\nImage resized to 224×224")
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN — Tabs

with right_col:
    tab_pred, tab_info = st.tabs(["🧪 Prediction", "ℹ️ Information"])

    # TAB 1 — Prediction

    with tab_pred:
        if uploaded_file is not None:   # Check if a file has been uploaded.
            image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format.

            # Image Preview (small height)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">🖼 MRI Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Preprocessing the image for prediction
            img = image.resize((224, 224))  # Resize to match model input size.
            img_array = np.array(img) / 255.0  # Normalize pixel values.
            img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension.

            # Prediction
            preds = model.predict(img_array)    # Get the model's predictions.
            predicted_class = class_names[np.argmax(preds)]   # Get the class with the highest probability.
            confidence = float(np.max(preds))

            # Prediction Result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">🧪 Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result">{predicted_class}</div>', unsafe_allow_html=True)
            st.progress(confidence)
            st.write(f"**Confidence: {confidence*100:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)

            # Probabilities
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">📊 Class Probabilities</div>', unsafe_allow_html=True)

            for i, prob in enumerate(preds[0]):     # Loop through each class and its probability.
                st.write(f"**{class_names[i]}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.2f}%")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("👈 Upload an MRI image to view predictions")
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2 — Information

    with tab_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 About Alzheimer’s Disease")
        st.write(
            "Alzheimer’s disease is a progressive neurological disorder that "
            "leads to memory loss and cognitive decline."
        )

        st.markdown("### 🏷 MRI Classification Classes")
        st.markdown("""
        - **Non Demented** – Healthy brain, No signs of Alzheimer’s detected. 
        - **Very Mild Dementia** – Earliest detectable stage, often with minor memory issues.  
        - **Mild Dementia** – Noticeable impairment, memory loss and cognitive decline.
        - **Moderate Dementia** – Significant memory impairment and daily functioning issues.
        """)

        st.markdown("### 🤖 Model Details")
        st.markdown("""
        - **Architecture:** ResNet50 (Transfer Learning)  
        - **Framework:** TensorFlow / Keras  
        - **Classes:** 4  
        - **Validation Accuracy:** ~90%
        """)

        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning(
            "This application is for **educational and research purposes only**. "
            "It is not a substitute for professional medical diagnosis."
        )

        st.markdown("— **Developed by Mohd Sami**")
        st.markdown('</div>', unsafe_allow_html=True)