import numpy as np
from random import randint

# Brillo a la imagen
def brightness(img):
    brightness = randint(70, 140)
    contrast = 30
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    dst = np.uint8(img)

    # amount = randint(30, 40)
    # a = np.double(img)
    # b = a + amount
    # dst = np.uint8(b)


    return dst