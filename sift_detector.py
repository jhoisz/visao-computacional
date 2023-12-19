import numpy as np
import cv2 as cv

def siftDetector(path):
    # Leitura e conversão da imagem para escala de cinza
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Criação do detector SIFT
    sift = cv.SIFT_create()

    # Detecção dos pontos de interesse
    kp = sift.detect(gray,None)

    # Desenho dos pontos de interesse na imagem
    img=cv.drawKeypoints(gray,kp,img)
    # img= cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite ( 'sift_keypoints.jpg' ,img)

    cv.imwrite('sift_keypoints.jpg',img)
    cv.imshow('SIFT Detector Result', img)