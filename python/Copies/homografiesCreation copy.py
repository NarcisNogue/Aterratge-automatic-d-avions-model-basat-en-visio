from ICGCService import ICGCService
import cv2
import math
import numpy as np

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
    def __init__(self, coordinates, show_image_level, width, length, image_side=520):
        self.SHOW_IMAGES_LEVEL = show_image_level
        self.LENGTH_PISTA = length
        self.WIDTH_PISTA = width
        self.coordinates = coordinates
        self.image_side = image_side
        self.cache = {}

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
        result_image = np.zeros((image_side,image_side,3))





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


        result = cv2.warpPerspective(image, h, (image_side,image_side))

        #Pintar horitzó
        result[horizon_mask, 0] = 255
        result[horizon_mask, 1] = 255
        result[horizon_mask, 2] = 0

        if(self.SHOW_IMAGES_LEVEL > 1):
            cv2.imshow("ImageInit2", result)

        # Importar la resta d'imatges TODO: Modificar perque sigui mes rapid i no importi una imatge per pixel en la llunyania
        for i in range(image_side):
            for j in range(image_side): # Double nested for loop yaay
                if(not result[i, j, :].any()):
                    res = np.dot(h_inv, np.array([j, i, 1]).reshape(3,1))


                    quadrant = np.array([np.floor((res[1]/res[2]) / image_side), np.floor((res[0]/res[2]) / image_side)]).astype(int).flatten()
                    new_image_coords = [
                        min_lat - vertical_margin + image_size*-quadrant[0],
                        min_lon - horiz_margin + image_size*quadrant[1],
                        max_lat + vertical_margin + image_size*-quadrant[0],
                        max_lon + horiz_margin + image_size*quadrant[1]
                    ]
                    new_image = np.zeros((image_side, image_side, 3), np.int8)
                    if(quadrant[0] in self.cache.keys() and quadrant[1] in self.cache[quadrant[0]].keys()):
                        new_image = self.cache[quadrant[0]][quadrant[1]]
                    else:
                        new_image = service.getSatImage(
                                new_image_coords[0],
                                new_image_coords[1],
                                new_image_coords[2],
                                new_image_coords[3],
                                height=image_side,
                                width=image_side,
                                layer="orto25c2016"
                            )
                        if(not quadrant[0] in self.cache.keys()):
                            self.cache[quadrant[0]] = {}
                        self.cache[quadrant[0]][quadrant[1]] = new_image


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

                    warped_mask[horizon_mask == 1] = 0

                    result[warped_mask == 1] = warped_new_image[warped_mask == 1]

                    if(self.SHOW_IMAGES_LEVEL > 0):
                        cv2.imshow("Loading...", result)
                        if cv2.waitKey(1) == ord('q'):
                            break
                    
        
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
        ], 3, 11.74, 50)
    resultImage, coords = homographyService.createHomography([-5, -30, 15], [-10, 0, 0])