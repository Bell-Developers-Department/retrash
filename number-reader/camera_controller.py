import cv2 as cv
import math


class CameraController():
    def __init__(self, _camera_number: int):
        self.camera_number = _camera_number

    def open_camera_and_wait_key(self, key: str = 't'):
        capture = cv.VideoCapture(self.camera_number)
        photo = None
        
        while True:
            is_recording, frame = capture.read()

            if not is_recording:
                break

            frame = self.place_square_at_center(frame, 100)
            cv.imshow('Camara', frame)

            key = cv.waitKey(25) & 0xFF
            if key == ord('t'):
                photo = self.take_pixels_from_square(frame, 100)
                break

        return photo


    def get_center_point(self, image_shape):
        return (math.ceil(float(image_shape[0] / 2)), math.ceil(float(image_shape[1] / 2)))
    
    def generate_square_cords(self, square_size, center_point):
        rectangle_from = (
            int(center_point[1] - square_size / 2),
            int(center_point[0] - square_size / 2)
        )

        rectangle_to = (
            int(center_point[1] + square_size / 2),
            int(center_point[0] + square_size / 2)
        )

        return (rectangle_from, rectangle_to)

    def place_square_at_center(self, image, square_size):
        center_point = self.get_center_point(image.shape)

        rectangle_from, rectangle_to = self.generate_square_cords(
            square_size, center_point)

        return cv.rectangle(
            image, rectangle_from, rectangle_to, (0, 0, 255), 2)
    
    def take_pixels_from_square(self, image, square_size):
        center_point = self.get_center_point(image.shape)

        rectangle_from, rectangle_to = self.generate_square_cords(
            square_size, center_point)

        return image[rectangle_from[1] + 2:rectangle_to[1] - 2, rectangle_from[0] + 2:rectangle_to[0] - 2]
