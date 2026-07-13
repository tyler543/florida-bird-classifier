import socket
import threading
import time
import json

from birdlib.local_store import get_pending, mark_synced
from birdlib.ebird import get_species_info, get_taxonomy, get_bird_id
from birdlib.supabase import insert_sighting


def check_wifi():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except Exception:
        return False


def _sync_one(row):
    predicted_species = row["predicted_species"]
    top_5 = json.loads(row["top_5"]) if row["top_5"] else {}

    ebird_info = get_species_info(predicted_species)
    taxonomy = get_taxonomy(predicted_species)
    bird_id = get_bird_id(ebird_info["species_code"]) if ebird_info else None

    sensor = {
        "lat": row["lat"],
        "lon": row["lon"],
        "fix": row["gps_fix"],
        "temp_f": row["temp_f"],
        "hum": row["humidity"],
        "batt_v": row["batt_v"],
        "utc": row["sensor_utc"],
    }

    insert_sighting(
        predicted_species=predicted_species,
        confidence=row["confidence"],
        top_5=top_5,
        ebird_info=ebird_info,
        taxonomy=taxonomy,
        bird_id=bird_id,
        sensor=sensor,
    )


def sync_pending():
    rows = get_pending()
    if not rows:
        return

    print(f"Syncing {len(rows)} pending sighting(s) to Supabase...")
    for row in rows:
        try:
            _sync_one(row)
            mark_synced(row["id"])
        except Exception as e:
            print(f"Sync failed for row {row['id']}: {e}")
            break


def _sync_loop():
    while True:
        time.sleep(30)
        if check_wifi():
            sync_pending()


def start_sync_thread():
    threading.Thread(target=_sync_loop, daemon=True).start()
