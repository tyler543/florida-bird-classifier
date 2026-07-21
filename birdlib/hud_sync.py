import threading
import time

from config import DEVICE_ID, HUD_SYNC_INTERVAL
from birdlib.supabase import get_user_from_device, get_hud_settings
from birdlib.overlay import send_hud_config


def _hud_sync_loop():
    user_id = get_user_from_device(DEVICE_ID)
    if user_id is None:
        print(f"No user mapped to device {DEVICE_ID} — HUD sync disabled")
        return

    last_version = None
    while True:
        hud = get_hud_settings(user_id)
        if hud:
            version = hud.get("hud_version")
            if version != last_version:
                last_version = version
                settings = hud["settings"]
                send_hud_config(
                    layout=settings.get("hud_layout", "layout1"),
                    color=settings.get("hud_color", "#FF0000"),
                    fields=settings.get("hud_fields"),
                )
                print(f"HUD config synced from Supabase (version {version})")
        time.sleep(HUD_SYNC_INTERVAL)


def start_hud_sync():
    threading.Thread(target=_hud_sync_loop, daemon=True).start()
