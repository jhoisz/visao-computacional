import numpy as np
import cv2 as cv

def siftDetector(path):
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)

    img=cv.drawKeypoints(gray,kp,img)
    # img= cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite ( 'sift_keypoints.jpg' ,img)

    cv.imwrite('sift_keypoints.jpg',img)
    cv.imshow('SIFT Detector Result', img)