import numpy as np
import cv2 as cv

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
    kp = sift.detect(gray,None)
    qtd_pontos_sift = len(kp)
    img=cv.drawKeypoints(gray,kp,img)
    print(qtd_pontos_sift)

    cv.imshow('SIFT', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # ORB
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures =5000)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    qtd_pontos_orb = len(kp)
    img = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    print(qtd_pontos_orb)

    cv.imshow('ORB', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
