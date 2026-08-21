import numpy as np
import streamlit as st
from PIL import Image

# Load model - cached so it only loads once
@st.cache_resource
def load_trained_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("mnist_ann.h5")
    return model

# Preprocess uploaded PIL image
def preprocess_image(image: Image.Image):
    image = image.convert('L').resize((28, 28))   # Grayscale + resize
    img_array = np.array(image) / 255.0           # Normalize to [0,1]
    img_array = img_array.reshape(1, 28, 28)      # Reshape for model
    return img_array

# Streamlit UI
def main():
    st.title("🔢 MNIST Digit Recognition")
    st.write("Upload a handwritten digit image to classify it.")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        with st.spinner("Classifying..."):
            model = load_trained_model()
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

        st.success(f"Predicted Digit: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
