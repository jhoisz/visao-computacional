import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def orbDetector(path):
    # Leitura e conversão da imagem para escala de cinza
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Criação do detector ORB
    orb = cv.ORB_create(nfeatures =300)

     # Detecção dos pontos de interesse
    kp = orb.detect(gray,None)

    # Cálcuo dos descritores
    # kp, des = orb.compute(img, kp)

    # Desenho dos pontos de interesse na imagem
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

    cv.imwrite('orb_keypoints.jpg',img2)
    plt.imshow(img2), plt.title("ORB Descriptor"), plt.show()