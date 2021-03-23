import tensorflow as tf
import numpy as np
import cv2
import time

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

imagebgr = cv2.imread("../../Datasets/BlenderDataset/Images/Image0045.png")

print(imagebgr.shape)


image = np.array([normalize(cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB))])

print(image.shape)

model = tf.keras.models.load_model('./Models/ModelTestBlender3.h5')

total_time = 0.0
iters = 100
for i in range(iters):
    start = time.time()
    pred_mask = model.predict(image)
    total_time += time.time() - start
print(total_time / iters)

cv2.imshow("Imatge", imagebgr)
cv2.imshow("Mask", pred_mask[0])
cv2.waitKey()