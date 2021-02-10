import numpy as np
import cv2
from skimage.draw import line_aa
import sys


if(len(sys.argv) < 3):
    exit()

cap = cv2.VideoCapture(sys.argv[1])

testFile = open(sys.argv[2], "r")

def paintCurrPoint(curr_point, frame):
    rr, cc, val = line_aa(curr_point[1], curr_point[0], curr_point[3], curr_point[2])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[3], curr_point[2], curr_point[5], curr_point[4])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[5], curr_point[4], curr_point[7], curr_point[6])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[7], curr_point[6], curr_point[1], curr_point[0])
    frame[rr,cc,2] = 255
    return frame

while(1):
    ret,frame = cap.read()
    if(not ret):
            break
    point = testFile.readline()
    frame_paint = paintCurrPoint([int(p) for p in point.split(",")], frame)

    cv2.imshow("frame", frame_paint)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break