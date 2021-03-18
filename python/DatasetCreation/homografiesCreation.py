from ICGCService import ICGCService
import cv2
import math
import numpy as np

service = ICGCService()

#########################################################################################################
################################### VARIABLES GLOBALS ###################################################
#########################################################################################################

SHOW_IMAGES_LEVEL = 2

LENGTH_PISTA = 50 #m
WIDTH_PISTA = 11.74 #m

MARGIN = 0.1
COORDS = np.array([
            [41.62778728171866, 2.2506523362405804], #A prop dreta
            [41.62782537455772, 2.250786446689412], #A prop esquerra
            [41.62692006354912, 2.251237060701778], #Lluny dreta
            [41.626875454201034, 2.2510969152827487] #Lluny esquerra
        ])

# Rotar la imatge de manera que encari la pista d'aterratge

# rot_angle = - math.atan2(COORDS[0][0] - COORDS[3][0], COORDS[3][1] - COORDS[0][1])*180/math.pi - 90
# print(rot_angle)

#########################################################################################################
################################### FUNCIONS GLOBALS ####################################################
#########################################################################################################
def findIntersection(p1,p2,p3,p4):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]
    if(( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) == 0 or ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) == 0):
        return False
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]
#########################################################################################################
################################### CALCULS INICIALS ####################################################
#########################################################################################################

max_lat = max(COORDS[:,0])
max_lon = max(COORDS[:,1])

min_lat = min(COORDS[:,0])
min_lon = min(COORDS[:,1])

image_side = 520

image_size = max((max_lon - min_lon),(max_lat - min_lat))

vertical_margin = (image_size - (max_lat - min_lat))/2
horiz_margin = (image_size - (max_lon - min_lon))/2

#### GET CORNER COORDS IN IMAGE
image_coords = np.zeros(COORDS.shape)
image_coords[:,1] = (image_size - (COORDS[:,0] - min_lat + vertical_margin)) / image_size * image_side
image_coords[:,0] = (COORDS[:,1] - min_lon + horiz_margin) / image_size * image_side
image_coords = image_coords.astype(int)
print(image_coords)


##### GET IMAGE
print(max_lat, min_lon)

image = service.getSatImage(
            min_lat - vertical_margin,
            min_lon - horiz_margin,
            max_lat + vertical_margin,
            max_lon + horiz_margin,
            height=image_side,
            width=image_side,
            layer="orto25c2016"
            )

# Marcar les cantonades

if(SHOW_IMAGES_LEVEL > 1):
    show_image = image.copy()
    for corner in image_coords:
        show_image = cv2.circle(show_image, (corner[0], corner[1]), 5, (0,255,0), -1)
    cv2.imshow("ImageInit", show_image)



#########################################################################################################
####################################### HOMOGRAFIA ######################################################
#########################################################################################################
result_image = np.zeros((image_side,image_side,3))





################################    POSICIONAR LA CAMERA EN EL MON ################################
## CONSIDERO COORDENADES (0,0,0) LA CANTONADA 0 DE LA PISTA

## CONFIG CAMERA ##
MAX_ANGLE = 30 #º Cap a cada banda (60 en total)

COORDS_PISTA_MON = np.array([
    [0,0,0],
    [-WIDTH_PISTA, 0, 0],
    [-WIDTH_PISTA, LENGTH_PISTA, 0],
    [0, LENGTH_PISTA, 0]
])

pos = (-30, -50, 30) #m (x, y, z)

angle = (-30, 0, 20) #º (de moment ignoraré el y... TODO i guess)

# GET COORDINATES DE LES CANTONADES
angles_cantonades = np.zeros((4,2))

for cant, angle_c in zip(COORDS_PISTA_MON, angles_cantonades):
    angle_c[0] = math.atan2(-cant[2] + pos[2], cant[1] - pos[1])*180/math.pi + angle[0]
    angle_c[1] = math.atan2(cant[0] - pos[0], cant[1] - pos[1])*180/math.pi - angle[2]

print(angles_cantonades)
target_points = np.zeros((4,2)).astype(int)

for target, angle_c in zip(target_points, angles_cantonades):
    target[0] = int((image_side / 2) + (image_side / 2)/MAX_ANGLE*angle_c[1])
    target[1] = int((image_side / 2) + (image_side / 2)/MAX_ANGLE*angle_c[0])

print(target_points)
print()

h, status = cv2.findHomography(image_coords, target_points)
h_inv = np.linalg.inv(h)

## GET HORIZON
# horizon_level = image_side - int(image_side/2 - (angle[0]/MAX_ANGLE*image_side))
cantonades_finals = cv2.perspectiveTransform(np.array([np.array([[0,0],[0, image_side-1],[image_side-1, image_side-1],[image_side-1, 0]]).astype(np.float32)]), h)[0]
# print(cantonades_finals)
# pendents = []
# for i in range(4):
#     pendents.append((cantonades_finals[(i+1)%4][1] - cantonades_finals[i][1])/(cantonades_finals[(i+1)%4][0] - cantonades_finals[i][0]))
# print(pendents)

max_Y = float('inf')
## Interseccions Rectes verticals (1->2 i 0->3)
point_ver = findIntersection(cantonades_finals[1], cantonades_finals[2], cantonades_finals[0], cantonades_finals[3])
if(point_ver is not False):
    min_Y = point_ver[1]

## Interseccions Rectes horitzontals (0->1 i 3->2)
point_hor = findIntersection(cantonades_finals[0], cantonades_finals[1], cantonades_finals[3], cantonades_finals[2])
if(point_hor is not False and point_hor[1] < min_Y):
    min_Y = point_ver[1]

horizon_level = int(max(0, min_Y) + image_side*0.15) # Una mica de marge

horizon_mask = np.zeros((image_side, image_side), np.uint8)

horizon_mask[0:horizon_level, :] = 1

print(horizon_level)



result = cv2.warpPerspective(image, h, (image_side,image_side))

#Pintar horitzó
result[0:horizon_level, :, 0] = 255
result[0:horizon_level, :, 1] = 255

cv2.imshow("ImageResult2", result)
# cv2.waitKey()

# Importar la resta d'imatges
for i in range(image_side):
    for j in range(image_side): # Double nested for loop yaay
        if(not result[i, j, :].any()):
            print(i, j)
            res = np.dot(h_inv, np.array([j, i, 1]).reshape(3,1))


            print(np.array([(res[0]/res[2]), (res[1]/res[2])]))
            quadrant = np.array([np.floor((res[1]/res[2]) / image_side), np.floor((res[0]/res[2]) / image_side)]).astype(int).flatten()
            print(quadrant)
            new_image_coords = [
                min_lat - vertical_margin + image_size*-quadrant[0],
                min_lon - horiz_margin + image_size*quadrant[1],
                max_lat + vertical_margin + image_size*-quadrant[0],
                max_lon + horiz_margin + image_size*quadrant[1]
            ]
            print(new_image_coords)
            print(min_lat - vertical_margin, min_lon - horiz_margin, max_lat + vertical_margin, max_lon + horiz_margin)
            new_image = service.getSatImage(
                    new_image_coords[0],
                    new_image_coords[1],
                    new_image_coords[2],
                    new_image_coords[3],
                    height=image_side,
                    width=image_side,
                    layer="orto25c2016"
                )


            translation_matrix = np.array(
                [
                    [1, 0, image_side*quadrant[1]],
                    [0, 1, image_side*quadrant[0]],
                    [0, 0, 1]
                ]
            )

            # cv2.imshow("ImageResult4", new_image)

            mask = np.ones((image_side, image_side), np.uint8)

            warped_new_image = cv2.warpPerspective(new_image, np.dot(h, translation_matrix), (image_side, image_side))
            warped_mask = cv2.warpPerspective(mask, np.dot(h, translation_matrix), (image_side, image_side))

            cv2.imshow("mask", warped_mask*255)

            warped_mask[horizon_mask == 1] = 0

            result[warped_mask == 1] = warped_new_image[warped_mask == 1]

            cv2.imshow("Result", result)
            if cv2.waitKey(1) == ord('q'):
                exit()
            







if(SHOW_IMAGES_LEVEL > 0):
    for corner in target_points:
        result = cv2.circle(result, (corner[0], corner[1]), 5, (0,255,0), -1)
    cv2.imshow("ImageResult1", result)
    cv2.waitKey()

