import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy


# data, metadata = tfds.load('mnist', as_supervised=True, with_info=True)

# training_data, testing_data = data['train'], data['test']


# def normalize(images, tags):
#     images = tf.cast(images, tf.float32)
#     images /= 255
#     return images, tags


# testing_data = testing_data.map(normalize).cache()

model = tf.keras.models.load_model('models/number-reader')

# print(model.evaluate(testing_data))

photo = cv2.imread('foto.jpg')
def divide_to_one(n): return n / 255


photo_mapped = divide_to_one(numpy.array(photo))
photo_tensor = tf.convert_to_tensor(photo_mapped, dtype=tf.float32)
photo_gray = tf.image.rgb_to_grayscale(photo_tensor)
photo_gray = tf.expand_dims(photo_gray, 0)
print(model.predict(photo_gray)[0].argmax())
