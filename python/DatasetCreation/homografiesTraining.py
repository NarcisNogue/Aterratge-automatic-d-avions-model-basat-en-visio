from ICGCService import ICGCService
import cv2
import math
import numpy as np
import imutils

service = ICGCService()

MARGIN = 0.1
COORDS = np.array([
            [41.62778728171866, 2.2506523362405804], #A prop dreta
            [41.62782537455772, 2.250786446689412], #A prop esquerra
            [41.62692779229079, 2.2512526679033935], #Lluny dreta
            [41.626889197692684, 2.2511192280068064] #Lluny esquerra
        ])

# Rotar la imatge de manera que encari la pista d'aterratge

rot_angle = - math.atan2(COORDS[0][0] - COORDS[3][0], COORDS[3][1] - COORDS[0][1])*180/math.pi - 90
print(rot_angle)


max_lat = max(COORDS[:,0])
max_lon = max(COORDS[:,1])

min_lat = min(COORDS[:,0])
min_lon = min(COORDS[:,1])

image_height = 520
image_width = (max_lon - min_lon)/(max_lat - min_lat)*image_height

margin_width = max((max_lon - min_lon), (max_lat - min_lat))*MARGIN

print(max_lat, min_lon)

image = service.getSatImage(
            min_lat - margin_width,
            min_lon - margin_width,
            max_lat + margin_width,
            max_lon + margin_width,
            height=image_height,
            width=image_width,
            layer="orto25c2012"
            )
cv2.imshow("Image", image)
# cv2.waitKey()

rotated = imutils.rotate_bound(image, rot_angle)
cv2.imshow("Image2", rotated)
cv2.waitKey()