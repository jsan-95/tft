import numpy as np
import cv2 as cv
import random
import decimal


# Ruido a la imagen (sal y pimienta)
def noise(img):
    amount = decimal.Decimal(random.randrange(40, 70)) / 100

    mean = 0
    var = 100
    sigma = var ** amount
    gaussian = np.random.normal(mean, sigma, (825, 758))  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv.normalize(noisy_image, noisy_image, 0, 220, cv.NORM_MINMAX, dtype=-1)
    dst = noisy_image.astype(np.uint8)

    return dst
