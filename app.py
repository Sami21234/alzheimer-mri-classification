import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image       

# App Config

st.set_page_config(
    page_title="Alzheimer MRI Detection",
    page_icon=":camera:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🧠 Alzheimer’s Disease Detection")
st.write("This app uses a deep learning model to detect Alzheimer's disease from MRI scans. Upload an MRI image to predict the stage of Alzheimer’s disease.")

# Now, loading the pre-trained model
@st.cache_resource    # Cache the model loading to improve performance
def load_model():     # function to load the model       
    model = tf.keras.models.load_model('models/best_resnet_model.h5')
    return model
model = load_model()    

# Class labels for the model(must be same as in the training phase)
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']

# Now, uploading the MRI image
uploaded_file = st.file_uploader("Upload MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:   # Check if a file has been uploaded
    #Now, Reading the image
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    
    # Preprocess the image for prediction
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Now, Making the prediction
    predictions = model.predict(image_array)    # Get the model's predictions
    predicted_class = class_names[np.argmax(predictions)]   # Get the class with the highest probability
    confidence = np.max(predictions) * 100    # Get the confidence of the prediction
    
    # Now, Outputting the prediction results
    st.markdown("### 🩺 Prediction Result")
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")