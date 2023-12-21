import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def orbDescriptor(path1, path2):
    # Leitura e conversão da imagem para escala de cinza
    img1 = cv.imread(path1)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.imread(path2)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Criação do detector ORB
    orb = cv.ORB_create()

    # Detecção dos pontos de interesse e calculo dos descritores
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Matcher criado para realizar a correspondência entre os descritores
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Realiza a correspondência entre os descritores
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # encontra os pontos de interese
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    fundamental_matrix, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
    print(len(matches))
    #desenha as linhas entre cada par de pontos na imagem
    for i in range(points1.shape[0]):
        if mask[i] != 0:
            pt1 = tuple(map(int, points1[i]))
            pt2 = tuple(map(int, points2[i]))
            cv.line(img1, pt1, pt2, (0,255,0), 1)
    cv.imwrite('orb_matches.jpg',img1)
    cv.imshow('Matching Points', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

