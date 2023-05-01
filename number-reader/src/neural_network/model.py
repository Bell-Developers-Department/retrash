from tensorflow import keras


class NumberReaderModel:
    def create_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, model, data, epochs=10, batch_size=32):
        model.fit(data, epochs=epochs, batch_size=batch_size)

    def save_weights(self, model, path):
        model.save_weights(path)

    def load_weights(self, model, path):
        model.load_weights(path)

    def test_model(self, model, data):
        x_train, y_train = data
        return model.evaluate(x_train, y_train, batch_size=32)
