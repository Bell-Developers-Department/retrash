import tensorflow as tf
import tensorflow_datasets as tfds
import math

data, metadata = tfds.load('mnist', as_supervised=True, with_info=True)

training_data, testing_data = data['train'], data['test']


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


training_data = training_data.map(normalize)
testing_data = testing_data.map(normalize)

training_data = training_data.cache()
testing_data = testing_data.cache()

training_data_amount = metadata.splits['train'].num_examples
testing_data_amount = metadata.splits['test'].num_examples


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    training_data,
    epochs=60,
    steps_per_epoch=math.ceil(training_data_amount / 32)
)

model.evaluate(testing_data[0], testing_data[1], verbose=2)
