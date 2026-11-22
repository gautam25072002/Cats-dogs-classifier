
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "cats_dogs_mobilenetv2.keras"
)

st.title("🐱 Cats vs Dogs Classifier")

st.write(
    "Upload an image and our MobileNetV2 model "
    "will identify whether it contains a cat or dog."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32)

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        image = np.expand_dims(image, axis=0)

        probability = float(
            model.predict(image, verbose=0)[0][0]
        )

        if probability >= 0.5:
            st.success("Prediction: Dog 🐶")
            st.write(f"Confidence: {probability * 100:.2f}%")
        else:
            st.success("Prediction: Cat 🐱")
            st.write(f"Confidence: {(1-probability) * 100:.2f}%")
