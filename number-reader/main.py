from camera_controller import CameraController
import cv2 as cv
import tensorflow as tf
import numpy

controller = CameraController(0)
photo = controller.open_camera_and_wait_key()
photo = cv.resize(photo, (28, 28))
photo = cv.cvtColor(photo, cv.COLOR_BGR2RGB)
cv.imwrite('foto.jpg', photo)
photo_tensor = tf.convert_to_tensor(photo, dtype=tf.float32)
photo_gray = tf.image.rgb_to_grayscale(photo_tensor)
photo_gray = tf.expand_dims(photo_gray, 0)
model = tf.keras.models.load_model('models/number-reader')
print(list(model.predict(photo_gray)))
