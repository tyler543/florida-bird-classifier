from picamera2 import Picamera2

def init_camera(width=640, height=480):
    picam2 = Picamera2()

    picam2.configure(
        picam2.create_preview_configuration(
            main={
                "format": "RGB888",
                "size": (width, height)
            }
        )
    )

    picam2.start()

    return picam2

def capture_frame(picam2):
    return picam2.capture_array()

def stop_camera(picam2):
    picam2.stop()