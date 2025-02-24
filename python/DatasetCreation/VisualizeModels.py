from makeRandomHomography import RandomHomographyCreator as rhc
import tensorflow as tf
import numpy as np
import cv2
import time
from skimage.draw import line_aa
import os

IMAGE_SIZE = 128
PARTITIONS = 8

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

# Carregar model
curr_path = os.path.dirname(os.path.realpath(__file__))
model = tf.keras.models.load_model(curr_path + "/Models/ModelTestGran1.h5")


# Carregar serveis
services = []
names = []

pistes = open(curr_path + "/coords_pistes.txt", "r")
coords_pista = pistes.readline()
while coords_pista and not "#DESCARTATS" in coords_pista:
    coords = np.array(coords_pista.split("//")[0].replace(",","").split(), dtype=np.float64).reshape(4,2)
    services.append(rhc(coords, IMAGE_SIZE*2, PARTITIONS))
    names.append(coords_pista.split("//")[1].replace("\n","").replace("\r",""))
    coords_pista = pistes.readline()
pistes.close()


while(1):
    result, cords, horizon_mask = services[0].getRandomHomography()
    if(result is not None):
        print("HELLOOOOOOOOOO")
        image = cv2.cvtColor(cv2.resize(result, (128,128)), cv2.COLOR_BGR2RGB)
        pred_mask = create_mask(model.predict(normalize(image[tf.newaxis, ...])))

        cv2.imshow("Imatge",  cv2.resize(result, (256, 256)))
        cv2.imshow("Mask", cv2.resize(pred_mask.numpy().astype(np.uint8)*255, (256, 256)))
        key = cv2.waitKey()
        if key == ord("q"):
            break
    else:
        print("Error creating homography, skipping image")
