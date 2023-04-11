from tensorflow import keras

imported_model = keras.models.load_model('models/conversor')

celsius = float(input('Ingrese los grados celsius: '))

print(imported_model.predict([celsius]))
