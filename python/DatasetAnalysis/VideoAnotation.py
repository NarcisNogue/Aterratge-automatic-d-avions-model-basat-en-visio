# ALGORITME ADAPTAT DE https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

import numpy as np
import cv2
from skimage.draw import line_aa
import sys

CLICKED_POINTS = []

SELECTED_POINT = None

if(len(sys.argv) < 3):
    exit()

cap = cv2.VideoCapture(sys.argv[1])

def onClickGetPoints(event, x, y, flags, params):
    if(event == cv2.EVENT_LBUTTONDOWN):
        CLICKED_POINTS.append([x, y])
    if(event==cv2.EVENT_RBUTTONDOWN):
        print(x, y)

def paintCurrPoint(curr_point, frame):
    rr, cc, val = line_aa(curr_point[0][1], curr_point[0][0], curr_point[1][1], curr_point[1][0])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[1][1], curr_point[1][0], curr_point[2][1], curr_point[2][0])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[2][1], curr_point[2][0], curr_point[3][1], curr_point[3][0])
    frame[rr,cc,2] = 255
    rr, cc, val = line_aa(curr_point[3][1], curr_point[3][0], curr_point[0][1], curr_point[0][0])
    frame[rr,cc,2] = 255
    return frame

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
ret,frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = []

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
valid = False

output = open(sys.argv[2], "w")

while(1):
    if(not valid):
        if(len(CLICKED_POINTS) == 4):
            p0 = np.array([np.array(CLICKED_POINTS.copy()).astype(dtype=np.float32)])
            output.write(",".join([str(i) for i in p0[0].flatten().astype(np.int32)]) + "\n")
            CLICKED_POINTS = []
            valid = True

    if(valid):
        ret,frame = cap.read()
        if(not ret):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        old_gray = frame_gray.copy()
        p0 = p1.copy()

    if(valid):
        frame_paint = paintCurrPoint(p0[0].astype(np.int32), frame)
        output.write(",".join([str(i) for i in p0[0].flatten().astype(np.int32)]) + "\n")
    else:
        frame_paint = frame.copy()

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", onClickGetPoints)
    cv2.imshow("frame", frame_paint)

    k = cv2.waitKey(30) & 0xff
    if k  == ord('q'):
        break
    elif k == ord('p'):
        valid = not valid

    # Now update the previous frame and previous points

cv2.destroyAllWindows()
cap.release()
output.close()
