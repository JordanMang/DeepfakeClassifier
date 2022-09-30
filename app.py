!pip install cv2
from unittest import result
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import pickle
import joblib
from PIL import Image, ImageOps
import tensorflow as tf
import pandas as pd
import torch

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

def main():
    st.subheader("Deepfake Detection")

    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)

    if st.button("Process"):
        result_img,result_resized = detect_faces(our_image)
        st.image(result_resized)

        model_file = open("./content/deepfake7GOOG.pkl", "rb")
        loaded_model = joblib.load(model_file)
        model_file.close()
        yhat = loaded_model.predict(np.expand_dims(result_resized/255, 0))
        st.write(f'{yhat}')
        if yhat >= 0.5:
            st.write("Real")
        if yhat < 0.5:
            st.write("Deepfake")
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(90, 90))
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        resized = cv2.resize(faces, (256, 256))
    return img,resized 

if __name__ == '__main__':
        main()    
