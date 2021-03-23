import tensorflow as tf
import numpy as np
import cv2
import time

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

from skimage.draw import line_aa
import sys


if(len(sys.argv) < 2):
    exit()

cap = cv2.VideoCapture(sys.argv[1])
model = tf.keras.models.load_model('./Models/ModelTestBlender3.h5')

while(1):
    ret,frame = cap.read()
    if(not ret):
            break

    image = np.array([normalize(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (128,128)))])


    total_time = 0.0
    iters = 100
    pred_mask = model.predict(image)

    cv2.imshow("Imatge", image)
    cv2.imshow("Mask", pred_mask[0])
    key = cv2.waitKey()
    if key == ord("q"):
        break