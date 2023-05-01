from data import NumberReaderData
from model import NumberReaderModel

model_controller = NumberReaderModel()
model = model_controller.create_model()

model_controller.compile_model(model)
model_controller.load_weights(
    model, r'C:\Programming\Projects\Bell\retrash\number-reader\checkpoints\checkpoint')

data_controller = NumberReaderData()
testing_data = data_controller.generate_testing_data()

print(model_controller.test_model(model, testing_data))
