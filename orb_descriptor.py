import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def orbDescriptor(path1, path2):
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    fundamental_matrix, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)

    img_matches = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    img_matches[:img1.shape[0], :img1.shape[1]] = img1

    for i in range(points1.shape[0]):
        if mask[i] != 0:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt1 = tuple(map(int, points1[i]))
            pt2 = tuple(map(int, points2[i]))
            cv.line(img_matches, pt1, pt2, color, 1)

    cv.imshow('Matching Points', img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

