import cv2 as cv


class CameraController():
    def __init__(self, _camera_number: int):
        self.camera_number = _camera_number

    def open_camera(self):
        capture = cv.VideoCapture(self.camera_number, cv.IMREAD_GRAYSCALE)
        photo = None
        while True:
            is_recording, frame = capture.read()

            if not is_recording:
                break

            cv.imshow('Camara', frame)

            key = cv.waitKey(25) & 0xFF
            if key == ord('t'):
                photo = frame
                break

        return photo
