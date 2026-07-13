import threading
import time

from config import DEVICE_ID, HUD_SYNC_INTERVAL
from birdlib.supabase import get_user_from_device, get_hud_settings
from birdlib.overlay import send_hud_config


def _hud_sync_loop():
    user_id = None
    last_version = None

    while True:
        try:
            if user_id is None:
                user_id = get_user_from_device(DEVICE_ID)
                if user_id is None:
                    print(f"No user mapped to device {DEVICE_ID} — retrying...")
                    time.sleep(HUD_SYNC_INTERVAL)
                    continue

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
        except Exception as e:
            print(f"HUD sync error (no wifi?): {e}")
        time.sleep(HUD_SYNC_INTERVAL)


def start_hud_sync():
    threading.Thread(target=_hud_sync_loop, daemon=True).start()
