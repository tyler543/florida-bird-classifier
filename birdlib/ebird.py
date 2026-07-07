import requests
from config import EBIRD_API_KEY, SUPABASE_URL, SUPABASE_KEY
from birdlib.bird_codes import EBIRD_CODES


def get_taxonomy(species_key):
    url = f"{SUPABASE_URL}/rest/v1/taxonomy"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    params = {
        "key": f"eq.{species_key}",
        "select": "common_name,scientific_name,conservation_status",
        "limit": "1",
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return {
        "common_name": species_key.replace("_", " "),
        "scientific_name": "Unknown",
        "conservation_status": "Unknown",
    }

def get_bird_id(species_code):
    url = f"{SUPABASE_URL}/rest/v1/birds"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    params = {
        "species_code": f"eq.{species_code}",
        "select": "id",
        "limit": "1",
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200 and response.json():
        return response.json()[0]["id"]
    return None


def get_species_info(species_name):
    code = EBIRD_CODES.get(species_name)
    
    if code is None:
        print(f"No eBird code found for {species_name}")
        return None

    url = f"https://api.ebird.org/v2/ref/taxonomy/ebird?species={code}&fmt=json"
    headers = {"X-eBirdApiToken": EBIRD_API_KEY}
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data:
            bird = data[0]
            return {
                "common_name": bird["comName"],
                "scientific_name": bird["sciName"],
                "species_code": bird["speciesCode"],
                "order": bird["order"],
                "family": bird["familyComName"]
            }
        else:
            print(f"No data returned for {species_name}")
            return None
    else:
        print(f"eBird API error: {response.status_code}")
        return None