import tensorflow as tf
import cv2
import numpy as np
import base64

img_size_px = 70
categories = ['Dog', 'Cat']
model = tf.keras.models.load_model("64x3-CNN-grey-high-res.model")


def prepare(image_base64):
    np_arr = np.fromstring(base64.b64decode(image_base64), np.uint8)
    img_arr = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    resized_img_arr = cv2.resize(img_arr, (img_size_px, img_size_px))
    return resized_img_arr.reshape(-1, img_size_px, img_size_px, 1)


def generate_prediction(image_base64):
    prediction = model.predict([prepare(image_base64)])
    print(prediction[0][0])
    return {'prediction': categories[int(prediction[0][0])]}
