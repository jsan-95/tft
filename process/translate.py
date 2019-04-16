import cv2 as cv
from random import randint


# Transladar en el eje x hacia la derecha a la imagen
def translateXRight(img):
    amount = randint(40, 100)
    img = img[0:825, 0:758 - amount]

    dst = cv.copyMakeBorder(img, 0, 0, amount, 0, cv.BORDER_CONSTANT)

    return dst


# Transladar en el eje x hacia la izquierda a la imagen
def translateXLeft(img):
    amount = randint(40, 100)
    img = img[0:825, 0 + amount:758]
    dst = cv.copyMakeBorder(img, 0, 0, 0, amount, cv.BORDER_CONSTANT)

    return dst


# Transladar el eje y hacia arriba a la imagen
def translateYUp(img):
    amount = randint(40, 100)
    img = img[0 + amount:825, 0:758]
    dst = cv.copyMakeBorder(img, 0, amount, 0, 0, cv.BORDER_CONSTANT)

    return dst


# Transladar el eje y hacia abajo a la imagen
def translateYDown(img):
    amount = randint(40, 100)
    img = img[0:825 - amount, 0:758]
    dst = cv.copyMakeBorder(img, amount, 0, 0, 0, cv.BORDER_CONSTANT)

    return dst