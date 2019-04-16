import numpy as np
import cv2 as cv
from random import randint


# Mediacion a la imagen
def averaging(img):
    amount = randint(20, 40)
    kernel = np.ones((5, 5), np.float32) / amount
    dst = cv.filter2D(img, -1, kernel)

    return dst
