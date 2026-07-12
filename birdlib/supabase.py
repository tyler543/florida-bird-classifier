import requests
from config import SUPABASE_URL, SUPABASE_KEY

def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

def get_user_from_device(device_serial):
    url = f"{SUPABASE_URL}/rest/v1/device_codes"
    params = {
        "device_serial": f"eq.{device_serial}",
        "select": "user_id",
        "limit": "1",
    }
    response = requests.get(url, headers=_headers(), params=params)
    if response.status_code == 200 and response.json():
        return response.json()[0]["user_id"]
    return None

def get_hud_settings(user_id):
    url = f"{SUPABASE_URL}/rest/v1/user_hud_settings"
    params = {
        "user_id": f"eq.{user_id}",
        "select": "settings,hud_version",
        "limit": "1",
    }
    response = requests.get(url, headers=_headers(), params=params)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return None

def insert_sighting(predicted_species, confidence, top_5, ebird_info, taxonomy=None, bird_id=None, sensor=None):
    url = f"{SUPABASE_URL}/rest/v1/bird_sightings"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    data = {
        "predicted_species": predicted_species,
        "confidence": confidence,
        "top_5": top_5,
        "ebird_info": ebird_info,
        "common_name": taxonomy.get("common_name") if taxonomy else None,
        "bird_id": bird_id,
        "lat": sensor.get("lat") if sensor else None,
        "lon": sensor.get("lon") if sensor else None,
        "gps_fix": sensor.get("fix") if sensor else None,
        "temp_f": sensor.get("temp_f") if sensor else None,
        "humidity": sensor.get("hum") if sensor else None,
        "batt_v": sensor.get("batt_v") if sensor else None,
        "sensor_utc": sensor.get("utc") if sensor else None,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        print(f"Sighting saved to Supabase")
    else:
        print(f"Supabase error: {response.status_code} {response.text}")
