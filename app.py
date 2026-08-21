import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


# Load the trained model
def load_trained_model():
    return load_model("mnist_ann.h5")

# Preprocess uploaded PIL image
def preprocess_image(image: Image.Image):
    image = image.convert('L').resize((28, 28))  # Grayscale + resize
    image = img_to_array(image) / 255.0           # Normalize
    image = image.reshape(1, 28, 28)              # Reshape for model
    return image

# Streamlit UI
def main():
    st.title("MNIST Digit Recognition")
    st.write("Upload a handwritten digit image to classify it.")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)          # Open once, reuse
        st.image(image, caption="Uploaded Image", width=150)

        model = load_trained_model()
        processed_image = preprocess_image(image)  # Pass PIL image directly
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
