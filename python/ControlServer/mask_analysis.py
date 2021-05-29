import cv2
import numpy as np


class Analyzer():
    def __init__(self, image_size, threshold_pista, threshold_sky):
        self.image_size = image_size
        self.threshold_pista = threshold_pista
        self.threshold_sky = threshold_sky
        self.pista_history = []
        self.sky_history = []



        # Params for analysing
        self.kernel = np.zeros((5,5)).astype(np.uint8)
        cv2.circle(self.kernel, (2, 2), 2, 1, -1)
        self.iters = 1

    def getPista(self, mask):
        mask_pista = (np.abs(mask - 1) < self.threshold_pista).astype(np.uint8)
        self.pista_history.insert(0, mask_pista)

        if(len(self.pista_history) > 10):
            self.pista_history.pop()

        # Netejar imatge de soroll
        mask_pista = cv2.erode(mask_pista, self.kernel, iterations = self.iters)
        mask_pista = cv2.dilate(mask_pista, self.kernel, iterations = self.iters)

        mask_pista = cv2.dilate(mask_pista, self.kernel, iterations = 1)
        mask_pista = cv2.erode(mask_pista, self.kernel, iterations = 1)



        # Buscar punt mig
        contours, hierarchy = cv2.findContours(mask_pista, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mean_x = 0
        mean_y = 0
        if(len(contours) > 0):
            max_contour = max(contours, key = cv2.contourArea)

            M = cv2.moments(max_contour )
            mean_x = int(M["m10"] / M["m00"])
            mean_y = int(M["m01"] / M["m00"])

            min_y = np.min(max_contour[:,0,1])
            max_y = np.max(max_contour[:,0,1])

            margin_near = 20
            margin_far = 8

            min_x_near = np.min(max_contour[:, 0, 0][np.abs(max_contour[:,0,1] - max_y) < margin_near])
            max_x_near = np.max(max_contour[:, 0, 0][np.abs(max_contour[:,0,1] - max_y) < margin_near])

            min_x_far = np.min(max_contour[:, 0, 0][np.abs(max_contour[:,0,1] - min_y) < margin_far])
            max_x_far = np.max(max_contour[:, 0, 0][np.abs(max_contour[:,0,1] - min_y) < margin_far])

            landing_x = (min_x_near + max_x_near) / 2

            cv2.circle(mask_pista, (np.uint8(np.ceil(mean_x)), np.uint8(np.ceil(mean_y))), 1, 2, 3)

            cv2.circle(mask_pista, (np.uint8(np.ceil(min_x_near)), np.uint8(np.ceil(max_y))), 1, 2, 3)
            cv2.circle(mask_pista, (np.uint8(np.ceil(max_x_near)), np.uint8(np.ceil(max_y))), 1, 2, 3)

            cv2.circle(mask_pista, (np.uint8(np.ceil(min_x_far)), np.uint8(np.ceil(min_y))), 1, 2, 3)
            cv2.circle(mask_pista, (np.uint8(np.ceil(max_x_far)), np.uint8(np.ceil(min_y))), 1, 2, 3)

            cv2.circle(mask_pista, (np.uint8(np.ceil(landing_x)), np.uint8(np.ceil(max_y))), 1, 2, 3)

            cv2.imshow("pista", cv2.resize(mask_pista.astype(np.uint8)*125, (256,256)))
            return landing_x, max_y
        else:
            cv2.imshow("pista", cv2.resize(mask_pista.astype(np.uint8)*125, (256,256)))    
            return None
        


