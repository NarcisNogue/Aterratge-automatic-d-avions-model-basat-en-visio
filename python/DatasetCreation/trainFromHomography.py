from skimage.draw import polygon2mask, polygon, polygon_perimeter
from homografiesCreation import HomographyCreator as hc
from random import uniform
import numpy as np
import math
import cv2

coords = [
            [41.62778728171866, 2.2506523362405804], #A prop dreta
            [41.62782537455772, 2.250786446689412], #A prop esquerra
            [41.62692006354912, 2.251237060701778], #Lluny dreta
            [41.626875454201034, 2.2510969152827487] #Lluny esquerra
        ]
length = 50
width = 11.74

service = hc(coords, 1, width, length, image_side=520)

base_x = -width/2
base_y = -20
base_z = 15

baseA_x = -10

radius = 50

rangeX = 10
rangeY = 20
rangeZ = 5

rangeAX = 5
rangeAZ = 5

while True:
    pos = [base_x + uniform(-rangeX, rangeX), base_y + uniform(-rangeY, rangeY), base_z + uniform(-rangeZ, rangeZ)]
    angle = [baseA_x + uniform(-rangeAX, rangeAX), 0, uniform(-rangeAZ, rangeAZ)]
    print(pos, angle)
    result, cords = service.createHomography(pos, angle)
    print(cords)
    mask = np.transpose(polygon2mask(result.shape[:-1], cords))

    cv2.imshow("Mask", mask.astype(np.uint8)*255)
    masked_result = result.copy()
    masked_result[mask] = (masked_result[mask] * 0.5).astype(np.uint8)
    masked_result[mask, 2] = 255
    cv2.imshow("Result with mask", masked_result)
    cv2.imshow("Result", result)
    if cv2.waitKey() == ord('q'):
        break