import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


model_path = "model.pkl" 
pipeline = joblib.load(model_path)


image_size = (50, 50)

def preprocess_image(image):
    image_resized = cv2.resize(image, image_size)
    image_normalized = image_resized / 255.0
    image_flatten = image_normalized.flatten().reshape(1, -1)
    return image_flatten


def main():
    st.title("Cat vs Dog Image Classification")
    st.write("Upload an image and the model will classify it as a cat or a dog.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_array = np.array(image)
        processed_image = preprocess_image(image_array)

        prediction = pipeline.predict(processed_image)
        prediction_prob = pipeline.predict_proba(processed_image)

        if prediction == 0:
            st.write("The model predicts this is a **Cat**. {:.2f}%.".format(prediction_prob[0][0]*100))
        else:
            st.write("The model predicts this is a **Dog**.  {:.2f}%.".format(prediction_prob[0][1]*100))

if __name__ == '__main__':
    main()
