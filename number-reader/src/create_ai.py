from neural_network.data import NumberReaderData
from neural_network.model import NumberReaderModel

model_controller = NumberReaderModel()
model = model_controller.create_model()
model_controller.compile_model(model)
model_controller.load_weights(
    model, r'C:\Programming\Projects\Bell\retrash\number-reader\checkpoints\checkpoint')
data_controller = NumberReaderData()
# training_data = data_controller.generate_training_data()
testing_data = data_controller.generate_testing_data()
print(model_controller.test_model(model, testing_data))
# model_controller.train_model(model, training_data, epochs=30)

# model_controller.save_weights(
#     model, r'C:\Programming\Projects\Bell\retrash\number-reader\checkpoints\checkpoint')
