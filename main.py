import os
import time
import queue
import threading
import torch
import cv2 as cv
from collections import deque

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
from birdlib.camera import init_camera, capture_frame, stop_camera
from birdlib.button import init_button
from gpiozero import LED as GpioLED
from birdlib.results import print_results, send_results
from birdlib.BLE import start_background, get_snapshot
from birdlib.overlay import send_live_frame, send_clear
from birdlib.hud_sync import start_hud_sync
from birdlib.local_store import init_db
from birdlib.sync import start_sync_thread


def _detect_motion_bbox(mask, shape):
    h, w = shape[:2]
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv.contourArea)
        if cv.contourArea(largest) > 500:
            x, y, bw, bh = cv.boundingRect(largest)
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(w - x, bw + pad * 2)
            bh = min(h - y, bh + pad * 2)
            return (x, y, bw, bh)
    bw, bh = 300, 220
    return (w // 2 - bw // 2, h // 2 - bh // 2, bw, bh)


def _crop_bbox(frame, x, y, w, h):
    h_f, w_f = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_f, x + w)
    y2 = min(h_f, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


button = init_button(BUTTON_GPIO)
LED = GpioLED(24)

classes = load_classes("classes.txt")
num_classes = len(classes)

torch.set_num_threads(2)

net, model_info = load_model(MODEL_NAME, MODEL_PATH, num_classes, DEVICE)
transform, size = build_transform(model_info)

bg_sub = cv.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
tracker = None
frame_counter = 0

prob_deque = deque(maxlen=5)
cached_species = None
cached_confidence = 0.0
cached_top5 = {}

box_x, box_y, box_w, box_h = 0, 0, 300, 220

collecting = False

_display_lock = threading.Lock()
_display_buf = None
_display_event = threading.Event()

def _frame_writer():
    while True:
        _display_event.wait()
        _display_event.clear()
        with _display_lock:
            buf = _display_buf
        if buf is not None:
            with open("/tmp/latest_frame_tmp.raw", "wb") as f:
                f.write(buf)
            os.replace("/tmp/latest_frame_tmp.raw", "/tmp/latest_frame.raw")

threading.Thread(target=_frame_writer, daemon=True).start()

_infer_q = queue.Queue(maxsize=1)

def _infer_worker():
    global cached_species, cached_confidence, cached_top5
    while True:
        crop = _infer_q.get()
        probs = run_inference(crop, net, transform, DEVICE)
        prob_deque.append(probs.cpu())
        avg_probs = average_frames(list(prob_deque))
        top_probs, top_indices = extract_topk_and_normalize(avg_probs, TOP_N)
        species, confidence, top5 = get_top_predictions(top_probs, top_indices, classes, TOP_N)
        if species != "unknown" and confidence >= CONF_THRESHOLD:
            cached_species = species
            cached_confidence = confidence
            cached_top5 = top5
            print(f"Live avg: {species} ({confidence:.2f})")

threading.Thread(target=_infer_worker, daemon=True).start()

init_db()
start_sync_thread()
start_background()
picam2 = init_camera()
start_hud_sync()

while True:
    frame = capture_frame(picam2)
    half_bgr = cv.resize(frame, (960, 540))
    with _display_lock:
        _display_buf = cv.cvtColor(half_bgr, cv.COLOR_BGR2RGB).tobytes()
    _display_event.set()

    small = cv.resize(half_bgr, (480, 270))
    bg_mask = bg_sub.apply(small)
    button_held = button.is_pressed

    if button_held and not collecting:
        collecting = True
        frame_counter = 0
        prob_deque.clear()
        cached_species = None
        cached_confidence = 0.0
        cached_top5 = {}
        LED.on()
        send_clear()


        raw_bbox = _detect_motion_bbox(bg_mask, small.shape)
        bbox_half = tuple(v * 2 for v in raw_bbox)
        try:
            tracker = cv.TrackerCSRT_create()
        except AttributeError:
            tracker = cv.legacy.TrackerCSRT_create()
        tracker.init(half_bgr, bbox_half)
        box_x, box_y, box_w, box_h = tuple(v * 2 for v in bbox_half)
        print("Button held — tracking started...")

    elif button_held and collecting:
        frame_counter += 1

        ok, bbox_half = tracker.update(half_bgr)
        if ok:
            bx, by, bw, bh = [int(v) for v in bbox_half]
            box_x, box_y, box_w, box_h = bx*2, by*2, bw*2, bh*2
        else:
            h_f, w_f = frame.shape[:2]
            box_x, box_y, box_w, box_h = w_f // 2 - 150, h_f // 2 - 110, 300, 220

        if frame_counter % INFERENCE_SKIP == 0:
            crop = _crop_bbox(frame, box_x, box_y, box_w, box_h)
            if crop is not None:
                try:
                    _infer_q.put_nowait(crop)
                except queue.Full:
                    pass

        if cached_species:
            send_live_frame(cached_species, cached_confidence, x=box_x, y=box_y, w=box_w, h=box_h)

    elif not button_held and collecting:
        collecting = False
        tracker = None
        LED.off()

        if prob_deque:
            avg_probs = average_frames(list(prob_deque))
            top_probs, top_indices = extract_topk_and_normalize(avg_probs, TOP_N)
            predicted_species, confidence, top5 = get_top_predictions(top_probs, top_indices, classes, TOP_N)

            if predicted_species != "unknown" and confidence >= CONF_THRESHOLD:
                print_results(predicted_species, confidence, top5, TOP_N)
                send_results(predicted_species, confidence, top5, sensor=get_snapshot(), bbox=(box_x, box_y, box_w, box_h))
                send_clear()
            else:
                print("No confident prediction — try holding longer or pointing more directly at the bird")
                send_clear()
        else:
            print("No inference ran — try holding longer")
            send_clear()

# cleanup
stop_camera(picam2)
cv.destroyAllWindows()
