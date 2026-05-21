import requests
from config import SUPABASE_URL, SUPABASE_KEY

def insert_sighting(predicted_species, confidence, top_5, ebird_info):
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
        "ebird_info": ebird_info
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        print(f"Sighting saved to Supabase")
    else:
        print(f"Supabase error: {response.status_code} {response.text}")