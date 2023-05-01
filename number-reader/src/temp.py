import cv2 as cv
import tensorflow as tf
from neural_network.model import NumberReaderModel
from neural_network.data import NumberReaderData
import numpy as np

photo = cv.imread(
    r'C:\Programming\Projects\Bell\retrash\number-reader\src\foto.jpg')
photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
photo = cv.resize(photo, (28, 28), interpolation=cv.INTER_NEAREST)
thr, photo = cv.threshold(photo, 100, 255, cv.THRESH_BINARY_INV)

cv.imwrite(
    r'C:\Programming\Projects\Bell\retrash\number-reader\src\foto-changed.jpg', photo)
photo = photo.astype('float32') / 255
photo = np.expand_dims(photo, axis=0)
print(photo)


model_controller = NumberReaderModel()
model = model_controller.create_model()
model_controller.compile_model(model)
model_controller.load_weights(
    model, r'C:\Programming\Projects\Bell\retrash\number-reader\checkpoints\checkpoint')

# data_controller = NumberReaderData()
# testing_data = data_controller.generate_testing_data()

# print(model_controller.test_model(model, testing_data))
print(model.predict(photo).argmax())
