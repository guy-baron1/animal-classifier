import tensorflow as tf
import cv2
import numpy as np
import base64

img_size_px = 120
categories = ['Dog', 'Cat']
model = tf.keras.models.load_model("64x3-CNN-grey-high-res.model")


def prepare(image_base64):
    np_arr = np.fromstring(base64.b64decode(image_base64), np.uint8)
    img_arr = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    resized_img_arr = cv2.resize(img_arr, (img_size_px, img_size_px))
    return resized_img_arr.reshape(-1, img_size_px, img_size_px, 1)


def generate_prediction(image_base64):
    prediction = model.predict([prepare(image_base64)])
    prediction_index = tf.argmax(prediction, axis=-1, output_type=tf.int32)
    prediction_label = categories[prediction_index[0]]
    print(prediction)
    return {'prediction': prediction_label, 'confidence': f'{prediction[0][prediction_index[0]] * 100}%'}
