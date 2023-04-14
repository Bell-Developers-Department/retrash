import cv2 as cv
import sys
import math


def get_center_point(image_shape):
    return (math.ceil(float(image_shape[0] / 2)), math.ceil(float(image_shape[1] / 2)))


def generate_square_cords(square_size, center_point):
    rectangle_from = (
        int(center_point[1] - square_size / 2),
        int(center_point[0] - square_size / 2)
    )

    rectangle_to = (
        int(center_point[1] + square_size / 2),
        int(center_point[0] + square_size / 2)
    )

    return (rectangle_from, rectangle_to)


def place_square_at_center(image, square_size):
    center_point = get_center_point(image.shape)

    rectangle_from, rectangle_to = generate_square_cords(
        square_size, center_point)

    print(rectangle_from)
    print(rectangle_to)

    return cv.rectangle(
        image, rectangle_from, rectangle_to, (0, 0, 255), 3)


img = cv.imread('saul-goodman.jpg', cv.IMREAD_GRAYSCALE)

if img is None:
    sys.exit("Could not read the image.")


img = cv.resize(img, (400, 400))
img = place_square_at_center(img, 200)

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img[100:300, 100:300])
