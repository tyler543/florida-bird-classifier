import socket
import json

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_UDP_IP = "127.0.0.1"
_UDP_PORT = 4242


def send_detection(common_name, scientific_name, conservation_status, confidence, x=60, y=60, w=300, h=220):
    payload = {
        "type": "detection",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "common_name": common_name,
        "scientific_name": scientific_name,
        "conservation_status": conservation_status,
        "confidence": confidence,
    }
    try:
        _sock.sendto(json.dumps(payload).encode("utf-8"), (_UDP_IP, _UDP_PORT))
    except Exception as e:
        print(f"Overlay UDP error: {e}")


def send_clear():
    try:
        _sock.sendto(json.dumps({"type": "clear"}).encode("utf-8"), (_UDP_IP, _UDP_PORT))
    except Exception as e:
        print(f"Overlay UDP error: {e}")


def send_hud_config(layout="layout1", color="#FF0000", fields=None):
    if fields is None:
        fields = ["Common Name", "Scientific Name", "Confidence", "Conservation Status"]
    payload = {"type": "hud", "hud_layout": layout, "hud_color": color, "hud_fields": fields}
    try:
        _sock.sendto(json.dumps(payload).encode("utf-8"), (_UDP_IP, _UDP_PORT))
    except Exception as e:
        print(f"Overlay UDP error: {e}")


def send_live_frame(predicted_species, confidence, x=60, y=60, w=300, h=220):
    payload = {
        "type": "detection",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "common_name": predicted_species.replace("_", " "),
        "scientific_name": "",
        "conservation_status": "",
        "confidence": confidence,
    }
    try:
        _sock.sendto(json.dumps(payload).encode("utf-8"), (_UDP_IP, _UDP_PORT))
    except Exception as e:
        print(f"Overlay UDP error: {e}")
