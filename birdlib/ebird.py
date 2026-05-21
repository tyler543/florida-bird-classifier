# birdlib/ebird.py
# Functions for interacting with the eBird API

import requests
from config import EBIRD_API_KEY
from birdlib.bird_codes import EBIRD_CODES

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