from picamera2 import Picamera2, Preview

def init_camera(width=1920, height=1080):
    picam2 = Picamera2()

    picam2.start_preview(Preview.DRM)
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)},
        lores={"format": "YUV420", "size": (640, 480)}
    ))
    picam2.start()
    return picam2

def capture_frame(picam2):
    return picam2.capture_array()

def capture_lores_frame(picam2):
    return picam2.capture_array("lores")

def stop_camera(picam2):
    picam2.stop_preview()
    picam2.stop()