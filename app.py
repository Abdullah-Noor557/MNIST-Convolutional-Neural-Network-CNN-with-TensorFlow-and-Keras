import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
    return image

model = load_model()

st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = preprocess_image(image)
    prediction = model.predict(image)
    st.write(f"Prediction: {np.argmax(prediction)}")
