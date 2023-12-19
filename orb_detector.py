import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def orbDetector(path):
    # Leitura e conversão da imagem para escala de cinza
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    # Criação do detector ORB
    orb = cv.ORB_create()

     # Detecção dos pontos de interesse
    kp = orb.detect(img,None)

    # Cálcuo dos descritores
    # kp, des = orb.compute(img, kp)

    # Desenho dos pontos de interesse na imagem
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

    cv.imwrite('orb_keypoints.jpg',img2)
    plt.imshow(img2), plt.title("ORB Descriptor"), plt.show()