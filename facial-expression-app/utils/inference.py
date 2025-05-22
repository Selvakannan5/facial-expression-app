import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/model.h5")
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def predict_expression(image):
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(1, 48, 48, 1) / 255.0
    predictions = model.predict(image)
    return labels[np.argmax(predictions)]