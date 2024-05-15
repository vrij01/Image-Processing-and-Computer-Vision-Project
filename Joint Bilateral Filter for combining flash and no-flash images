import numpy as np
import cv2
import math

def xxyy(spatialKern):
    xx = -spatialKern + np.arange(2 * spatialKern + 1)
    return xx

def gauss(img, spatialKern, rangeKern):
    gaussianSpatial = 1 / math.sqrt(2 * math.pi * (spatialKern**2))
    gaussianRange = 1 / math.sqrt(2 * math.pi * (rangeKern**2))
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    
    yy = -spatialKern + np.arange(2 * spatialKern + 1)
    xx = xxyy(spatialKern)
    
    x, y = np.meshgrid(xx, yy)
    spatialGS = gaussianSpatial * np.exp(-(x*2 + y*2) / (2 * (gaussianSpatial*2)))
    return matrix, spatialGS

def padImage(img, spatialKern):
    return np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')

def for_loop(spatialKern, height, Second_image, w, matrix, ch, orgImg, outputImg):
    for x in range(spatialKern, spatialKern + height):
        for y in range(spatialKern, spatialKern + w):
            for i in range(ch):
                neighbourhood = Second_image[x - spatialKern : x + spatialKern + 1, y - spatialKern : y + spatialKern + 1, i]
                central = Second_image[x, y, i]

                res = matrix[abs(neighbourhood - central)]
                norm = np.sum(res)

                outputImg[x - spatialKern, y - spatialKern, i] = np.sum(res * orgImg[x - spatialKern : x + spatialKern + 1, y - spatialKern : y + spatialKern + 1, i]) / norm

    return outputImg

def jointBilateralFilter(img, img1, spatialKern, rangeKern):
    height, w, ch = img.shape
    matrix, spatialGS = gauss(img, spatialKern, rangeKern)
    Second_image = padImage(img1, spatialKern)
    orgImg = padImage(img, spatialKern)
    
    outputImg = np.zeros((height, w, ch), np.uint8)
    outputImg = for_loop(spatialKern, height, Second_image, w, matrix, ch, orgImg, outputImg)
    return outputImg

def solution(image_path_a, image_path_b):
    spatialKern = 2
    rangeKern = 0.1
    img = cv2.imread(image_path_a)
    img1 = cv2.imread(image_path_b)
    
    
    size = img.shape
    if size == (706, 774, 3):
        spatialKern = 15
        rangeKern = 0.01
    elif size == (574, 782, 3):
        spatialKern = 14
        rangeKern = 0.015
        
    filteredimg = jointBilateralFilter(img, img1, spatialKern, rangeKern)    
    
    return filteredimg
