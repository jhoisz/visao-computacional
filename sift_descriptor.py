import cv2 as cv
import numpy as np

# Carregue as duas imagens
def siftDescriptor(path1, path2):
    # Leitura e conversão da imagem para escala de cinza
    img1 = cv.imread(path1)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.imread(path2)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Inicialize o detector e descritor SIFT
    sift = cv.SIFT_create()

    # Detecte pontos de interesse e calcule os descritores para ambas as imagens
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Inicialize o matcher de correspondência de características (usando o BFMatcher neste exemplo)
    bf = cv.BFMatcher()

    # Faça a correspondência entre os descritores das duas imagens
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Aplique o teste de razão para filtrar correspondências robustas
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extraia os pontos correspondentes
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Calcule a matriz de transformação fundamental usando o método RANSAC
    fundamental_matrix, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
    print("pei")
    # desenha as linhas entre cada par de pontos na imagem
    cont=0;
    for i in range(points1.shape[0]):
        if mask[i] != 0:
            cont+=1
            pt1 = tuple(map(int, points1[i]))
            pt2 = tuple(map(int, points2[i]))
            cv.line(img1, pt1, pt2, (0, 255, 0), 1)
    print(cont)
    cv.imwrite('sift_matches.jpg', img1)
    cv.imshow('Matching Points', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

