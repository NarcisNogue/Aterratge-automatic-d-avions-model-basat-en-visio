from homografiesCreation import HomographyCreator as hc
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

service = hc(coords, 0, width, length)

base_x = -width/2
base_y = length/2

radius = 50
alt = 15

for a in range(-5,5):
    pos = [base_x + a, -30, alt]
    angle = [-10, 0, 0]
    print(pos, angle)
    result, cords = service.createHomography(pos, angle)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord('q'):
        break