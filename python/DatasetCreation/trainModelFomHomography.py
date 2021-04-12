from skimage.draw import polygon2mask, polygon, polygon_perimeter
from makeRandomHomography import RandomHomographyCreator as rhc
import numpy as np
import random
import cv2
import os

NUM_IMAGES = 30
image_size = 128
partitions = 1

curr_path = os.path.dirname(os.path.realpath(__file__))
pistes = open(curr_path + "/coords_pistes.txt", "r")
coords_pista = pistes.readline()

services = []
names = []

while coords_pista and not "#DESCARTATS" in coords_pista:
    coords = np.array(coords_pista.split("//")[0].replace(",","").split(), dtype=np.float64).reshape(4,2)
    services.append(rhc(coords, image_size, partitions))
    names.append(coords_pista.split("//")[1].replace("\n","").replace("\r",""))
    coords_pista = pistes.readline()
pistes.close()


while True:
    images = []
    masks = []
    it = 0
    while it < NUM_IMAGES:
        index = random.choice(range(len(services)))
        result, cords, horizon_mask = services[index].getRandomHomography()

        if(result is not None):
            mask = np.transpose(polygon2mask(result.shape[:-1], cords))
            # cv2.imshow("True Mask", cv2.resize(mask.astype(np.uint8)*255, (256, 256)))
            # masked_result = result.copy()
            # masked_result[mask] = (masked_result[mask] * 0.5).astype(np.uint8)
            # masked_result[mask, 2] = 255
            # cv2.imshow("Result with mask", cv2.resize(masked_result, (256, 256)))
            # result = cv2.resize(result, (256, 256))
            # cv2.imshow("Result", cv2.putText(result,names[index], (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2))

            # key = cv2.waitKey()
            # if(key == ord("q")):
            #     break
            print("HELLOOOOOOOOOO")
            images.append(result)
            masks.append(mask)
            it += 1
        else:
            print("Error creating homography, skipping image")
    images = np.array(images)
    masks = np.array(masks)
    for im, mask in zip(images, masks):
        cv2.imshow("True Mask", cv2.resize(mask.astype(np.uint8)*255, (256, 256)))
        cv2.imshow("Result", cv2.resize(im, (256, 256)))
        cv2.waitKey()