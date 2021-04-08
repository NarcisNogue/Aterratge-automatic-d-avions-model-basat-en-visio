from skimage.draw import polygon2mask, polygon, polygon_perimeter
from homografiesCreation import HomographyCreator as hc
from random import uniform
# import tensorflow as tf
import numpy as np
import math
import cv2

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

coords = [
            [41.62778728171866, 2.2506523362405804], #A prop dreta
            [41.62782537455772, 2.250786446689412], #A prop esquerra
            [41.62692006354912, 2.251237060701778], #Lluny esquerra
            [41.626875454201034, 2.2510969152827487] #Lluny dreta
        ]
length = 50
width = 11.74

image_side = 128

strip_width = image_side*1.0 / length * width

service = hc(coords, 1, width, length, image_side=image_side, partitions=3)

# model = tf.keras.models.load_model('../DatasetAnalysis/Models/ModelTestBlender3.h5')

def getRandomHomography():
    t = image_side / 2
    
    rotation1 = np.random.randint(0,360)
    perpectiveAngle = np.random.randint(0,60)
    scale = 1 - max(np.abs(np.random.normal())/3, .8)
    translation = [np.random.randint(-t,t), np.random.randint(-t,t)]
    rotation2 = np.random.normal()/3*180
    target_cords = np.array(
                [
                    [int(image_side/2 + strip_width/2), image_side],
                    [int(image_side/2 - strip_width/2), image_side],
                    [int(image_side/2 - strip_width/2), 0],
                    [int(image_side/2 + strip_width/2), 0]
                ])

    # Rotation 1
    angle_rot1 = np.radians(rotation1)
    rotation_m = np.array([
        [math.cos(angle_rot1), -math.sin(angle_rot1), -t*math.cos(angle_rot1)+t*math.sin(angle_rot1)+t],
        [math.sin(angle_rot1), math.cos(angle_rot1), -t*math.sin(angle_rot1)-t*math.cos(angle_rot1)+t],
        [0, 0, 1]
    ])
    target_cords = np.dot(rotation_m, np.r_[target_cords.transpose(), [[1,1,1,1]]]).transpose().astype(int)[:,:2]

    # Perspective, scale and translation
    for cord in target_cords:
        cord[1] = (cord[1] - t)*math.cos(np.radians(perpectiveAngle)) + t
        cord[0] = (((cord[0] - t)*(cord[1] + 10) * .05 + t) - cord[0]) * math.sin(np.radians(perpectiveAngle)) + cord[0]

        cord[0] = (cord[0] - t) * scale + t + translation[0]
        cord[1] = (cord[1] - t) * scale + t + translation[1]

    result, cords = service.createHomography(None, [None, rotation2, None], target_cords)

    return result, cords

while True:
    result, cords = getRandomHomography()
    # image = np.zeros((0, 128,128,3))
    # for i in [x/10.0 for x in range(3, 18, 2)]:
    #     result2 = result*i
    #     result2[result2 > 255] = 255
    #     result2 = result2.astype(np.uint8)
    #     image = np.vstack((image, np.array([normalize(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))])))
    
    # pred_mask = model.predict(image)

    mask = np.transpose(polygon2mask(result.shape[:-1], cords))

    # mean_pred_mask = np.mean(pred_mask, axis=0)
    # cv2.imshow("Predicted Mask", mean_pred_mask)
    cv2.imshow("True Mask", cv2.resize(mask.astype(np.uint8)*255, (256, 256)))
    masked_result = result.copy()
    masked_result[mask] = (masked_result[mask] * 0.5).astype(np.uint8)
    masked_result[mask, 2] = 255
    cv2.imshow("Result with mask", cv2.resize(masked_result, (256, 256)))
    cv2.imshow("Result", cv2.resize(result, (256, 256)))

    key = cv2.waitKey()
    if(key == ord("q")):
        break
    print("HELLOOOOOOOOOO")
