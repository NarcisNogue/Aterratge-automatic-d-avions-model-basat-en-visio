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

service = hc(coords, 1, width, length, image_side=256)

pos_x = -width/2
pos_y = -20
pos_z = 15

angle_x = -10
angle_y = 0
angle_z = 0

delta_pos = 5
delta_angle = 2

while True:
    pos = [pos_x, pos_y, pos_z]
    angle = [angle_x, angle_y, angle_z]
    result, cords = service.createHomography(pos, angle)
    mask = np.transpose(polygon2mask(result.shape[:-1], cords))

    cv2.imshow("Mask", mask.astype(np.uint8)*255)
    masked_result = result.copy()
    masked_result[mask] = (masked_result[mask] * 0.5).astype(np.uint8)
    masked_result[mask, 2] = 255
    cv2.imshow("Result with mask", masked_result)
    cv2.imshow("Result", result)


    key = cv2.waitKey()
    if(key == ord('q')):
        angle_y += delta_angle
    elif(key == ord('e')):
        angle_y -= delta_angle

    elif(key == ord('i')):
        angle_x += delta_pos
    elif(key == ord('k')):
        angle_x -= delta_pos

    elif(key == ord('j')):
        angle_z -= delta_pos
    elif(key == ord('l')):
        angle_z += delta_pos


    # Les posicions son en coordenades globals, ja ho cambiare
    elif(key == ord('w')):
        pos_y += delta_pos*math.cos(np.radians(angle_z))
        pos_x += delta_pos*math.sin(np.radians(angle_z))
    elif(key == ord('s')):
        pos_y -= delta_pos*math.cos(np.radians(angle_z))
        pos_x -= delta_pos*math.sin(np.radians(angle_z))

    elif(key == ord('a')):
        pos_x -= delta_pos*math.cos(np.radians(angle_z))
        pos_y += delta_pos*math.sin(np.radians(angle_z))
    elif(key == ord('d')):
        pos_x += delta_pos*math.cos(np.radians(angle_z))
        pos_y -= delta_pos*math.sin(np.radians(angle_z))

    elif(key == ord(' ')):
        pos_z += delta_pos
    elif(key == ord('z')): #Shift
        pos_z -= delta_pos

    
    #Tecles especials
    elif(key == 27): #ESC
        break