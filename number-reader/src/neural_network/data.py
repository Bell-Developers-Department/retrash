from tensorflow import keras


class NumberReaderData:
    def augment_data(self, data, batch_size=32):
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            zoom_range=[0.5, 1.5])

        x_data, y_data = data

        data_generator.fit(x_data)

        return data_generator.flow(x_data, y_data, batch_size=batch_size)

    def generate_training_data(self):
        (x_training_data, y_training_data), * \
            _ = keras.datasets.mnist.load_data()

        x_training_data = x_training_data.reshape(
            x_training_data.shape[0], 28, 28, 1)

        x_training_data = x_training_data.astype('float32') / 255

        y_training_data = keras.utils.to_categorical(y_training_data)

        augmented_training_data = self.augment_data(
            (x_training_data, y_training_data))

        return augmented_training_data

    def generate_testing_data(self):
        _, (x_testing_data, y_testing_data) = keras.datasets.mnist.load_data()

        x_testing_data = x_testing_data.reshape(
            x_testing_data.shape[0], 28, 28, 1)

        x_testing_data = x_testing_data.astype('float32') / 255

        y_testing_data = keras.utils.to_categorical(y_testing_data)

        return x_testing_data, y_testing_data
