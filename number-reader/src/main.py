from recorder import camera, screen_tools
from neural_network.model import NumberReaderModel
from adapters import adapt_photo

camera_controller = camera.CameraController()
video_capture = camera_controller.record_camera(0)
photo = None
while True:
    is_recording, frame = camera_controller.read_frame(video_capture)

    if not is_recording:
        break

    frame_with_square = screen_tools.place_square_at_center(frame, 100)
    camera_controller.show_frame('Cámara', frame_with_square)

    def take_photo():
        photo = screen_tools.take_pixels_from_square(frame, 100)
        break

    camera_controller.wait_key('t', take_photo)

photo_adapted = adapt_photo(photo)

model_controller = NumberReaderModel()
model = model_controller.create_model()
model_controller.compile_model(model)
model_controller.load_weights(
    model, r'C:\Programming\Projects\Bell\retrash\number-reader\checkpoints\checkpoint')

print(model.predict(photo_adapted).argmax())
