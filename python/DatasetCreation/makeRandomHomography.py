from skimage.draw import polygon2mask, polygon, polygon_perimeter
from homografiesCreation import HomographyCreator as hc
from random import uniform
# import tensorflow as tf
import numpy as np
import math
import cv2

def getDistanceFromLatLonInM(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in m
    dLat = np.radians(lat2-lat1)
    dLon = np.radians(lon2-lon1) 
    a = math.sin(dLat/2.0) * math.sin(dLat/2.0) + math.cos(np.radians(lat1)) * math.cos(np.radians(lat2)) * math.sin(dLon/2.0) * math.sin(dLon/2.0)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0-a)); 
    d = R * c; # Distance in km
    return d


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image



class RandomHomographyCreator:
    def __init__(self, coords, image_side, partitions):
        # coords = [
        #             [41.62778728171866, 2.2506523362405804], #A prop dreta
        #             [41.62782537455772, 2.250786446689412], #A prop esquerra
        #             [41.62692006354912, 2.251237060701778], #Lluny esquerra
        #             [41.626875454201034, 2.2510969152827487] #Lluny dreta
        #         ]
        length = getDistanceFromLatLonInM(coords[1][0], coords[1][1], coords[2][0], coords[2][1])
        width = getDistanceFromLatLonInM(coords[0][0], coords[0][1], coords[1][0], coords[1][1])

        self.image_side = image_side

        self.strip_width = image_side*1.0 / length * width

        self.service = hc(coords, 1, width, length, image_side=image_side, partitions=partitions)

    # model = tf.keras.models.load_model('../DatasetAnalysis/Models/ModelTestBlender3.h5')

    def getRandomHomography(self):
        t = self.image_side / 2
        
        rotation1 = np.random.normal()/6*180
        perpectiveAngle = np.random.randint(0,60)
        scale = 1 - max(np.abs(np.random.normal())/3, .8)
        translation = [np.random.normal()/6*t, np.random.normal()/6*t]
        rotation2 = np.random.normal()/10*180
        print(rotation2)
        target_cords = np.array(
                    [
                        [int(self.image_side/2 + self.strip_width/2), self.image_side],
                        [int(self.image_side/2 - self.strip_width/2), self.image_side],
                        [int(self.image_side/2 - self.strip_width/2), 0],
                        [int(self.image_side/2 + self.strip_width/2), 0]
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

        result, cords = self.service.createHomography(None, [None, rotation2, None], target_cords)

        return result, cords


if __name__ == "__main__":
    while True:

        coords = [
                    [41.62778728171866, 2.2506523362405804], #A prop dreta
                    [41.62782537455772, 2.250786446689412], #A prop esquerra
                    [41.62692006354912, 2.251237060701778], #Lluny esquerra
                    [41.626875454201034, 2.2510969152827487] #Lluny dreta
                ]

        randomHCreator = RandomHomographyCreator(coords, 520, 5)

        result, cords = randomHCreator.getRandomHomography()

        mask = np.transpose(polygon2mask(result.shape[:-1], cords))
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
