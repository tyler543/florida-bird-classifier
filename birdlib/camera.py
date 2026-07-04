from picamera2 import Picamera2, Preview
from libcamera import controls
from config import TUNING_FILE

def init_camera(width=1920, height=1080):
    tuning = Picamera2.load_tuning_file(TUNING_FILE) if TUNING_FILE else None
    picam2 = Picamera2(tuning=tuning)

    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)},
        #lores={"format": "YUV420", "size": (640, 480)}
    ))
    picam2.start_preview(Preview.DRM, x=0, y=0, width=width, height=height)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})
    return picam2

def capture_frame(picam2):
    return picam2.capture_array()

def capture_lores_frame(picam2):
    return picam2.capture_array("lores")

def stop_camera(picam2):
    picam2.stop_preview()
    picam2.stop()