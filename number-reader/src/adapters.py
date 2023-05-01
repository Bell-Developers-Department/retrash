import cv2 as cv
import numpy as np


def adapt_photo(photo):
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    photo_resized = cv.resize(photo_gray, (28, 28),
                              interpolation=cv.INTER_NEAREST)

    _, photo_thresholded = cv.threshold(
        photo_resized, 100, 255, cv.THRESH_BINARY_INV)

    photo_float = photo_thresholded.astype('float32') / 255
    photo_4d = np.expand_dims(photo_float, axis=0)

    return photo_4d
