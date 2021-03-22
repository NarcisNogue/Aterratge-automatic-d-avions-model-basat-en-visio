import cv2
import os
import numpy as np

folder = "../../Datasets/BlenderDataset/Masks/"
images = sorted(os.listdir(folder))

for image in images:
    mask = (cv2.imread(folder+image, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    print(mask.shape)
    # cv2.imshow("hola?", mask*255)
    # cv2.waitKey()
    cv2.imwrite(folder+image, mask)