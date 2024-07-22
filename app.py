import streamlit as st
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class names
class_names = ['cat', 'dog']

def predict_image(img):
    # Load and preprocess the image
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizing

    # Predict the class
    prediction = model.predict(img_array)
    return class_names[int(prediction[0][0] > 0.5)]

# Streamlit app
st.title("Cat or Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image(Cat or Dog ONLY"), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
    img = Image.open(uploaded_file)
    prediction = predict_image(img)
    
    # Display the result
    st.write(f"Prediction: {prediction}")
