from skimage.draw import polygon2mask, polygon, polygon_perimeter
from ICGCService import ICGCService
import cv2
import math
import numpy as np
import time

service = ICGCService()
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
    t = (((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)))/(((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))
    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    return [px, py]

def line_eq(p1, p2):
    m = ((p2[1] - p1[1])) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    return m, c
#########################################################################################################
################################### CALCULS INICIALS ####################################################
#########################################################################################################
class HomographyCreator:
    def __init__(self, coordinates, show_image_level, width, length, image_side=520, partitions=5):
        self.SHOW_IMAGES_LEVEL = show_image_level
        self.LENGTH_PISTA = length
        self.WIDTH_PISTA = width
        self.coordinates = coordinates
        self.image_side = image_side
        self.cache = {},
        self.partitions = partitions

    def createHomography(self, pos, angle, target_coordinates=None):
        COORDS = np.array(self.coordinates)
        max_lat = max(COORDS[:,0])
        max_lon = max(COORDS[:,1])

        min_lat = min(COORDS[:,0])
        min_lon = min(COORDS[:,1])

        image_side = self.image_side

        image_size = max((max_lon - min_lon),(max_lat - min_lat))

        vertical_margin = (image_size - (max_lat - min_lat))/2
        horiz_margin = (image_size - (max_lon - min_lon))/2

        #### GET CORNER COORDS IN IMAGE
        image_coords = np.zeros(COORDS.shape)
        image_coords[:,1] = (image_size - (COORDS[:,0] - min_lat + vertical_margin)) / image_size * image_side
        image_coords[:,0] = (COORDS[:,1] - min_lon + horiz_margin) / image_size * image_side
        image_coords = image_coords.astype(int)
        

        ##### GET IMAGE

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

        if(self.SHOW_IMAGES_LEVEL > 1):
            show_image = image.copy()
            for corner in image_coords:
                show_image = cv2.circle(show_image, (corner[0], corner[1]), 5, (0,255,0), -1)
            cv2.imshow("ImageInit", show_image)



        #########################################################################################################
        ####################################### HOMOGRAFIA ######################################################
        #########################################################################################################
        result = np.zeros((image_side,image_side,3), dtype=np.uint8)

        ################################    POSICIONAR LA CAMERA EN EL MON ################################
        ## CONSIDERO COORDENADES (0,0,0) LA CANTONADA 0 DE LA PISTA

        ## CONFIG CAMERA ##
        MAX_ANGLE = 30 #º Cap a cada banda (60 en total)

        COORDS_PISTA_MON = np.array([
            [0,0,0],
            [-self.WIDTH_PISTA, 0, 0],
            [-self.WIDTH_PISTA, self.LENGTH_PISTA, 0],
            [0, self.LENGTH_PISTA, 0]
        ])

        t = image_side/2
        if(target_coordinates is None):
            # GET COORDINATES DE LES CANTONADES
            angles_cantonades = np.zeros((4,2))
            distancia_focal = t / math.tan(np.radians(MAX_ANGLE))

            

            for cant, angle_c in zip(COORDS_PISTA_MON, angles_cantonades): # TODO: No calcula be la y quan esta darrere la camera / sota la pantalla
                angle_c[0] = np.degrees(math.atan2(cant[0] - pos[0], cant[1] - pos[1]))- angle[2] #X
                angle_c[1] = np.degrees(math.atan2((cant[2] + pos[2]), ((cant[1] - pos[1]) / math.cos(np.radians(angle_c[0]))))) + angle[0] #Y
            print(angles_cantonades[0][1])

            target_points = np.zeros((4,2)).astype(int)

            for target, angle_c in zip(target_points, angles_cantonades):
                target[0] = int(t + distancia_focal * math.tan(np.radians(angle_c[0])))
                target[1] = int(t + distancia_focal * math.tan(np.radians(angle_c[1])))
            print(target_points[0])
        else:
            target_points = target_coordinates

        angle_y = np.radians(angle[1])
        rotation_m = np.array([
            [math.cos(angle_y), -math.sin(angle_y), -t*math.cos(angle_y)+t*math.sin(angle_y)+t],
            [math.sin(angle_y), math.cos(angle_y), -t*math.sin(angle_y)-t*math.cos(angle_y)+t],
            [0, 0, 1]
        ])
        target_points = np.dot(rotation_m, np.r_[target_points.transpose(), [[1,1,1,1]]]).transpose().astype(int)[:,:2]
        h, status = cv2.findHomography(image_coords, target_points)
        h_inv = np.linalg.inv(h)

        h_image_to_world, status2 = cv2.findHomography(target_points, COORDS[:,::-1])
        h_inv_image_to_world = np.linalg.inv(h_image_to_world)

        ## GET HORIZON
        # horizon_level = image_side - int(image_side/2 - (angle[0]/MAX_ANGLE*image_side))
        cantonades_finals = cv2.perspectiveTransform(np.array([np.array([[0,0],[0, image_side-1],[image_side-1, image_side-1],[image_side-1, 0]]).astype(np.float32)]), h)[0]

        ## Interseccions Rectes verticals (1->2 i 0->3)
        point_1 = findIntersection(cantonades_finals[1], cantonades_finals[2], cantonades_finals[0], cantonades_finals[3])

        ## Interseccions Rectes horitzontals (0->1 i 3->2)
        point_2 = findIntersection(cantonades_finals[0], cantonades_finals[1], cantonades_finals[3], cantonades_finals[2])

        m, c = line_eq(point_1, point_2)

        #Mirar si el terra esta a sota o a sobre l'horitzó
        isUnder = (target_points[0][0]*m + c > target_points[0][1])*2 - 1
        c -= image_side*0.05*isUnder/math.cos(np.radians(angle[1]))

        horizon_mask = np.fromfunction(lambda i, j: isUnder*(m*j + c) < isUnder*i, (image_side, image_side), dtype=int)
        
        if(self.SHOW_IMAGES_LEVEL > 2):
            cv2.imshow("Horizon", horizon_mask.astype(np.uint8)*255)


        # result = cv2.warpPerspective(image, h, (image_side,image_side))

        #Pintar horitzó
        result[horizon_mask, 0] = 255
        result[horizon_mask, 1] = 255
        result[horizon_mask, 2] = 0

        if(self.SHOW_IMAGES_LEVEL > 1):
            cv2.imshow("ImageInit2", result)

        step = np.floor(image_side/float(self.partitions))
        for i in range(self.partitions + int(step*self.partitions < image_side)): # Afegeix una particio si cal per haver arrodonit avall
            for j in range(self.partitions + int(step*self.partitions < image_side)):
                start = time.time()
                min_x = step*i
                max_x = min(step*(i+1), image_side - 1)
                min_y = step*j
                max_y = min(step*(j+1), image_side - 1)

                square_points = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y] 
                ])

                end_points = np.zeros((0,2))
                for point in square_points:
                    if(isUnder * (point[0] * m + c) > isUnder * point[1]):
                        end_points = np.vstack((end_points, point))

                if(end_points.shape[0] == 4):
                    # print(end_points)
                    h_inv_points = np.dot(h_inv, np.r_[end_points.transpose(), [[1,1,1,1]]])
                    h_inv_points = h_inv_points / h_inv_points[2]
                    h_inv_points = h_inv_points.transpose()[:,:2]


                    mask42 = np.transpose(polygon2mask(result.shape[:-1], h_inv_points))
                    cv2.imshow("mask42", mask42.astype(np.uint8) * 255)

                    # FINS AQUI ESTA BE ----
                    
                    # print(h_inv_points)
                    # print()
                    
                    
                    # lat_lon_points = h_inv_points / (np.array([max_lat - min_lat, max_lon - min_lon]) / image_size * image_side) * image_size + np.array([min_lat, min_lon])

                    # lat_lon_points = h_inv_points[:,::-1] / image_side * image_size * np.array([1, -1]) + np.array([min_lat - vertical_margin, min_lon - horiz_margin])

                    lat_lon_points = np.dot(h_image_to_world, np.r_[end_points.transpose(), [[1,1,1,1]]])
                    lat_lon_points = lat_lon_points / lat_lon_points[2]
                    lat_lon_points = lat_lon_points.transpose()[:,:2][:,::-1]

                    # print(lat_lon_points)
                    
                    min_lat_curr = min(lat_lon_points[:,0])
                    min_lon_curr = min(lat_lon_points[:,1])

                    max_lat_curr = max(lat_lon_points[:,0])
                    max_lon_curr = max(lat_lon_points[:,1])

                    # max_min_coords_curr = np.array

        
                    new_image = service.getSatImage(
                        min_lat_curr,
                        min_lon_curr,
                        max_lat_curr,
                        max_lon_curr,
                        height=image_side,
                        width=image_side,
                        layer="orto25c2016"
                    )

                    cv2.imshow("Desc", new_image)
                    
                    # -- Fins aqui funciona
                    

                    # ----------------------------- IGNORAR
                    image_points = ((lat_lon_points - np.array([min_lat_curr, min_lon_curr]))/np.array([max_lat_curr-min_lat_curr, max_lon_curr-min_lon_curr])*image_side)
                    image_points[:,1] = (image_side - image_points[:,1])
                    # image_points[:,0] = (image_side - image_points[:,0])

                    
                    image_points_end = np.dot(h_inv_image_to_world, np.r_[lat_lon_points[:,::-1].transpose(), [[1,1,1,1]]])
                    image_points_end = image_points_end / image_points_end[2]
                    image_points_end = image_points_end.transpose()[:,:2]
                    # ------------------------------------------

                    cantonades_im_result = np.array([
                        [min_lat_curr, min_lon_curr],
                        [min_lat_curr, max_lon_curr],
                        [max_lat_curr, min_lon_curr],
                        [max_lat_curr, max_lon_curr]
                    ])

                    homography_points = np.dot(h_inv_image_to_world, np.r_[cantonades_im_result[:,::-1].transpose(), [[1,1,1,1]]])
                    homography_points = homography_points / homography_points[2]
                    homography_points = homography_points.transpose()[:,:2]
                    

                    result3 = new_image.copy()
                    for corner in image_points.astype(int):
                        result3 = cv2.circle(result3, (corner[0], corner[1]), 5, (0,255,0), -1)
                    cv2.imshow("ImageResult1", result3)

                    h2, status = cv2.findHomography(np.array([
                        [0, image_side],
                        [image_side, image_side],
                        [0,0],
                        [image_side, 0],
                    ]), homography_points)

                    mask = np.ones((image_side, image_side), np.uint8)

                    warped_new_image = cv2.warpPerspective(new_image, h2, (image_side, image_side))
                    warped_mask_total = cv2.warpPerspective(mask, h2, (image_side, image_side))
                    # warped_mask[:,:int(min_x)] = 0
                    # warped_mask[:,int(max_x):] = 0
                    # warped_mask[:int(min_y),:] = 0
                    # warped_mask[int(max_y):,:] = 0

                    warped_mask = np.transpose(polygon2mask(result.shape[:-1], image_points_end))

                    cv2.imshow("Masksksks", warped_mask.astype(np.uint8)*255)
                    cv2.imshow("MaskTotal", warped_mask_total.astype(np.uint8)*255)

                    warped_mask[horizon_mask == 1] = 0

                    result[warped_mask == 1] = warped_new_image[warped_mask == 1]

                    cv2.imshow("resulelslslslss", result)
                    # cv2.waitKey()
                    print(time.time() - start)




        if(self.SHOW_IMAGES_LEVEL > 1):
            result2 = result.copy()
            for corner in target_points:
                result2 = cv2.circle(result2, (corner[0], corner[1]), 5, (0,255,0), -1)
            cv2.imshow("ImageResult1", result2)
            cv2.waitKey()
        return result, target_points

if(__name__ == "__main__"):
    homographyService = HomographyCreator([
            [41.62778728171866, 2.2506523362405804], #A prop dreta
            [41.62782537455772, 2.250786446689412], #A prop esquerra
            [41.62692006354912, 2.251237060701778], #Lluny dreta
            [41.626875454201034, 2.2510969152827487] #Lluny esquerra
        ], 3, 11.74, 50, image_side=128)
    resultImage, coords = homographyService.createHomography([-5, -30, 15], [-10, 0, 0])