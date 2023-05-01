import cv2 as cv
import screen_tools as st


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

            frame = st.place_square_at_center(frame, 100)
            cv.imshow('Camara', frame)

            key = cv.waitKey(25) & 0xFF
            if key == ord('t'):
                photo = st.take_pixels_from_square(frame, 100)
                break

        return photo

    def record_camera(self, camera_number: int):
        return cv.VideoCapture(camera_number)

    def read_frame(self, capture):
        return capture.read()

    def show_frame(self, title, frame):
        cv.imshow(title, frame)

    def wait_key(self, expected_key, callback):
        pressed_key = cv.waitKey(25) & 0xFF

        if pressed_key == ord(expected_key):
            return callback()
