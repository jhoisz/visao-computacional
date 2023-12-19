import numpy as np
import cv2 as cv


def intersecKp(kp1, kp2, img):
    intersec = kp1.intersection(kp2)
    kp_intersec = [cv.KeyPoint(x, y, 1) for x, y in intersec]

    img = cv.drawKeypoints(img, kp_intersec, None, color=(0, 255, 0), flags=0)
    cv.imshow('Os dois acharam', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def newKp(kp1, kp2, img):
    diff = kp1 - kp2
    kp_diff = [cv.KeyPoint(x, y, 1) for x, y in diff]
    img = cv.drawKeypoints(img, kp_diff, None, color=(0, 255, 0), flags=0)
    cv.imshow('Só esse achou', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def comparison(path):
    # HARRIS
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)

    #dst = cv.dilate(dst,None)

    img[dst>0.01*dst.max()] = [0,0,255]

    points_of_interest = np.argwhere(img == [0,0,255])

    qtd_pontos_harris = len(points_of_interest)
    print(qtd_pontos_harris)
    cv.imshow('HARRIS', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # SIFT
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints_sift = sift.detect(gray,None)
    qtd_pontos_sift = len(keypoints_sift)
    coords_sift = set((int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints_sift)
    img=cv.drawKeypoints(gray,keypoints_sift,img)
    print(qtd_pontos_sift)

    cv.imshow('SIFT', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # ORB
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures =5000)
    keypoints_orb = orb.detect(img,None)
    keypoints_orb, des = orb.compute(img, keypoints_orb)
    qtd_pontos_orb = len(keypoints_orb)
    coords_orb = set((int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints_orb)

    img = cv.drawKeypoints(img, keypoints_orb, None, color=(0,255,0), flags=0)
    print(qtd_pontos_orb)

    cv.imshow('ORB', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    #apenas sift encontrou
    img = cv.imread(path)
    newKp(coords_sift, coords_orb, img)
    #apenas orb encontrou
    img = cv.imread(path)
    newKp(coords_orb, coords_sift, img)
    #ambos encontratam
    img = cv.imread(path)
    intersecKp(coords_sift, coords_orb, img)



