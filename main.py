import time
import cv2 as cv

from config import *
from birdlib.inference import (
    load_classes,
    load_model,
    build_transform,
    run_inference,
    average_frames,
    get_top_predictions,
    extract_topk_and_normalize,
)
from libcamera import controls
from birdlib.camera import init_camera, capture_frame, stop_camera, capture_lores_frame
from birdlib.button import init_button
from birdlib.results import print_results, send_results
from birdlib.BLE import start_background, get_snapshot
# variables
inference_hz = INFERENCE_HZ # inference per second
inference_interval = 1.0 / inference_hz
last_inference_time = 0.0
frames_probabilities = []
button = init_button(BUTTON_GPIO)
LED = init_button(24)

classes = load_classes("classes.txt")
num_classes = len(classes)

net, model_info = load_model(
    MODEL_NAME,
    MODEL_PATH,
    num_classes,
    DEVICE
)
transform, size = build_transform(model_info)

collecting = False
waiting_for_release = False

start_background()

picam2 = init_camera()

while True:
    frame = capture_frame(picam2) # full res display
    #lores_frame = capture_lores_frame(picam2) # low res for inference
    '''
    removed for Raspi OS LITE (no display)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    '''
    
    button_held = button.is_pressed 

    if waiting_for_release:
        if not button_held:
            waiting_for_release = False
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        continue
    
    elif button_held and not collecting:
        # button just pressed — lock focus on whatever is in frame
        collecting = True
        frames_probabilities = []
        picam2.set_controls({"AfMode": controls.AfModeEnum.Auto, "AfTrigger": controls.AfTriggerEnum.Start})
        time.sleep(0.3)
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": picam2.capture_metadata()["LensPosition"]})
        print("Button held — collecting frames...")

    elif button_held and collecting:
        # still holding, run inference on this frame
        now = time.perf_counter()
        if now - last_inference_time >= inference_interval:
            last_inference_time = now

            probs = run_inference(
                frame,
                net,
                transform,
                DEVICE
            )
            frames_probabilities.append(probs.cpu())

            print(f"Frame {len(frames_probabilities)}/{FRAME_AVERAGE_SIZE} captured")

            if len(frames_probabilities) >= FRAME_AVERAGE_SIZE:
                avg_probs = average_frames(frames_probabilities)
                
                avg_top_probs, avg_top_indices = extract_topk_and_normalize(
                    avg_probs,
                    TOP_N
                )
                 
                predicted_species, confidence, top_5 = get_top_predictions(
                    avg_top_probs,
                    avg_top_indices,
                    classes,
                    TOP_N
                )
               
                print_results(predicted_species, confidence, top_5, TOP_N)
                send_results(predicted_species, confidence, top_5, sensor=get_snapshot())

                collecting = False
                frames_probabilities = []
                waiting_for_release = True

    elif not button_held and collecting:
        # button released — return to continuous AF
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        if len(frames_probabilities) < FRAME_AVERAGE_SIZE:
            print(f"Button released too early — hold for the full 3 seconds ({FRAME_AVERAGE_SIZE} frames needed, got {len(frames_probabilities)})")
        collecting = False
        frames_probabilities = []
# cleanup
stop_camera(picam2)
cv.destroyAllWindows()