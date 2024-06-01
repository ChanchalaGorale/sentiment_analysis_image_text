import tensorflow as tf

import numpy as np

import cv2

def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

    return model

def preprocess_image(image_path):
    img= cv2.imread(image_path)
    img= cv2.resize(img, 224, 224)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def analyze_image_sentiment(image_path, model):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions









