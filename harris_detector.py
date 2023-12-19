import numpy as np
import cv2 as cv

def harrisDetector(path):

    # leitura e conversão da imagem para escala de cinza
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # conversão da imagem para float32 (array NumPy de 32 bits)
    gray = np.float32(gray)

    # detecção de cantos utilizando Harris com a imagem na escala de cinza
    dst = cv.cornerHarris(gray,2,3,0.04)

    # dilatação da imagem para melhor visualização dos cantos detectados
    dst = cv.dilate(dst,None)

    # marcação dos cantos detectados com um círculo vermelho
    img[dst>0.01*dst.max()]=[0,0,255]

    # Salva e exibe a imagem
    cv.imwrite('harris_keypoints.jpg',img)
    # cv.imshow('Harris Detector Result', img)

    
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()