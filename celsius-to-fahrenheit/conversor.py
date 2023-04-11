import tensorflow as tf
import numpy
keras = tf.keras


def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(units=3, input_shape=[1]),
        keras.layers.Dense(units=3),
        keras.layers.Dense(units=1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    return model


def train_model(model):
    celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    return model.fit(celsius, fahrenheit, epochs=1000, verbose=False)


def save_model(model, filename: str):
    model.save('models/' + filename)


created_model = create_model()
train_model(created_model)
save_model(created_model, 'conversor')
