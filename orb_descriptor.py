import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def orbDescriptor(path1, path2):
    # Leitura e conversão da imagem para escala de cinza
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)

    # Criação do detector ORB
    orb = cv.ORB_create()

    # Detecção dos pontos de interesse e calculo dos descritores
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Matcher criado para realizar a correspondência entre os descritores
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Realiza a correspondência entre os descritores
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenha os matches
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite('orb_matches.jpg',img_matches)
    plt.imshow(img_matches), plt.title("ORB Descriptor"), plt.show()

