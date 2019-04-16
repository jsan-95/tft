import cv2 as cv
from random import randrange


# Difuminado a la imagen
def blur(img):
    amount = randrange(5, 17, 2)
    dst = cv.GaussianBlur(img, (amount, amount), 0)

    return dst
