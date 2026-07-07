# cd "OneDrive\Desktop\florida-bird-classifier-main"
# .venv\Scripts\activate
# python webcam_birder.py
# (.venv) C:\Users\grace\OneDrive\Desktop\florida-bird-classifier-mainC:\Users\grace\.venv\Scripts\python.exe webcam_birder.py

DEVICE_ID = "FEATHER-000"

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY
url = SUPABASE_URL
key = SUPABASE_KEY

supabase = create_client(url, key)


import cv2
import birder
from birder.inference.classification import infer_image
from PIL import Image

import socket
import json
UDP_IP = "127.0.0.1"
UDP_PORT = 4242

import time
import threading
import random

from collections import deque
history = deque(maxlen=10)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------
# load bird database from supabase
# -----------------------
def get_bird_info(bird_key):
    print("LOOKUP KEY:", repr(bird_key))

    response = supabase.table("taxonomy") \
        .select("*") \
        .eq("key", bird_key) \
        .execute()

    data = response.data
    print("RAW RESPONSE:", data)

    if not data:
        return {
            "common_name": bird_key,
            "scientific_name": "Unknown",
            "conservation_status": "Unknown"
        }

    return data[0]

# -----------------------
# load device ids database from supabase
# -----------------------

def get_user_from_device(device_serial):
    response = (
        supabase.table("device_codes")
        .select("user_id")
        .eq("device_serial", device_serial)
        .single()
        .execute()
    )

    print("DEVICE LOOKUP:", response)

    if not response.data:
        print("No user mapped to device!")
        return None

    return response.data["user_id"]

# -----------------------
# load hud settings database from supabase
# -----------------------

def get_hud_settings(user_id):
    print(f"Looking up HUD settings for user: {user_id}")

    response = (
        supabase.table("user_hud_settings")
        .select("settings, hud_version")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    print("Response data:", response.data)

    if not response.data:
        print("No HUD settings found.")
        return None

    return {
        "settings": response.data["settings"],
        "hud_version": response.data["hud_version"]
    }

# -----------------------
# Load model
# -----------------------
model_name = "mobilenet_v4_s_il-common"

(net, model_info) = birder.load_pretrained_model(
    model_name,
    inference=True
)

size = birder.get_size_from_signature(model_info.signature)
transform = birder.classification_transform(size, model_info.rgb_stats)

idx_to_class = {v: k for k, v in model_info.class_to_idx.items()}

# -----------------------
# Open webcam
# -----------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

frame_count = 0

last_bird = "Searching..."
last_confidence = 0.0
info = {}

user_id = get_user_from_device(DEVICE_ID)

if user_id is None:
    print("No user assigned to device. Exiting.")
    exit()


hud = get_hud_settings(user_id)

if hud is None:
    hud = {
        "settings": {
            "hud_color": "#FFFFFF",
            "hud_layout": "layout1",
            "hud_fields": ["Common Name", "Scientific Name", "Confidence", None]
        },
        "hud_version": 0
    }
print("HUD LOADED:", hud)

last_hud_version = hud["hud_version"]
hud_settings = hud["settings"]


# -----------------------
# Update hud settings
# -----------------------

hud_state = None
last_hud_version = -1

def hud_sync_loop():
    global hud_state, last_hud_version
    
    while True:
        try:
            hud_check = get_hud_settings(user_id)
            print("HUD CHECK RAW:", hud_check)

            if not hud_check:
                time.sleep(2)
                continue
            
            version = int(hud_check.get("hud_version", -1))
            print("HUD VERSION:", hud_check.get("hud_version"))
            print("LAST VERSION:", last_hud_version)

            if version == last_hud_version:
                time.sleep(2)
                continue

            last_hud_version = version
            hud_state = hud_check

            hud_payload = {
                "type": "hud",
                "hud_color": hud_state["settings"]["hud_color"],
                "hud_fields": hud_state["settings"]["hud_fields"],
                "hud_layout": hud_state["settings"]["hud_layout"]
            }

            sock.sendto(
                json.dumps(hud_payload).encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )
                
            print("SENDING HUD PAYLOAD UDP:", hud_payload)

        except Exception as e:
            print("HUD SYNC ERROR:", e)

        time.sleep(2)

threading.Thread(target=hud_sync_loop, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    frame_count += 1

    if frame_count % 10 == 0:

        # -----------------------
        # RUN MODEL
        # -----------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        (out, _) = infer_image(net, image, transform)

        probs = out[0]
        best = probs.argmax()

        bird = idx_to_class.get(best)
        if not bird:
            continue
        confidence = float(probs[best])
        
        # -----------------------
        # SMOOTHING (no flicker)
        # -----------------------
        history.append((bird, confidence))

        counts = {}
        conf_sum = {}

        for b, c in history:
            counts[b] = counts.get(b, 0) + 1
            conf_sum[b] = conf_sum.get(b, 0) + c

        stable_bird = max(counts, key=counts.get)
        stable_conf = conf_sum[stable_bird]/counts[stable_bird] 

        bird_key = stable_bird.replace(" ", "_")
        print("LOOKUP KEY:", bird_key)

        last_bird = stable_bird
        last_confidence = stable_conf

        # -----------------------
        # SUPABASE LOOKUP 
        # -----------------------
        info = get_bird_info(bird_key)

        if "settings" not in hud:
            print("HUD ERROR: missing settings block")
            continue

        h_frame, w_frame = frame.shape[:2]

        # -----------------------------
        # Coordinates and Size
        # -----------------------------
        h_frame, w_frame = frame.shape[:2]

        padding = 40

        min_h = 50
        min_w = 280  # optional but recommended so labels fit horizontally

        max_w = min(280, w_frame - padding * 2)
        max_h = min(300, h_frame - padding * 2)

        w = random.randint(min_w, max_w)
        h = random.randint(min_h, max_h)

        max_x = w_frame - w - padding
        max_y = h_frame - h - padding

        x = random.randint(padding, max(0, max_x))
        y = random.randint(padding, max(0, max_y))
                
        
        payload = {
            "type": "detection",
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "common_name": info["common_name"].replace("_", " ").title(),
            "scientific_name": info["scientific_name"],
            "conservation_status": info["conservation_status"],
            "confidence": stable_conf,
        }

        sock.sendto(
            json.dumps(payload).encode("utf-8"),
            (UDP_IP, UDP_PORT)
        )

        print("SENDING PAYLOAD UDP:", payload)

    # -----------------------
    # LOCAL DISPLAY
    # -----------------------


    cv2.putText(display,
        f"{info.get('common_name', last_bird)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(display,
        f"Scientific: {info.get('scientific_name', 'Unknown')}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(display,
        f"Status: {info.get('conservation_status', 'Unknown')}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    cv2.putText(display,
        f"Confidence: {last_confidence:.2f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Birder Webcam", display)

    if cv2.waitKey(1) == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()