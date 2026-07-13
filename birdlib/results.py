from birdlib.ebird import get_species_info, get_taxonomy, get_bird_id
from birdlib.supabase import insert_sighting
from birdlib.overlay import send_detection
from birdlib.local_store import save_pending
from birdlib.sync import check_wifi


def print_results(predicted_species, confidence, top_5, top_n):
    print("\n--- Inference Result ---")
    print(f"Top {top_n} predictions:")
    for species, score in top_5.items():
        print(f"{species}: {score:.4f}")
    print(f"Predicted class: {predicted_species} with confidence {confidence:.4f}")


def send_results(predicted_species, confidence, top_5, sensor=None, bbox=None):
    if predicted_species == "unknown":
        print("Unknown species — skipping eBird and Supabase")
        return

    x, y, w, h = bbox if bbox else (60, 60, 300, 220)

    if check_wifi():
        try:
            ebird_info = get_species_info(predicted_species)
            taxonomy = get_taxonomy(predicted_species)
            bird_id = get_bird_id(ebird_info["species_code"]) if ebird_info else None

            insert_sighting(
                predicted_species=predicted_species,
                confidence=confidence,
                top_5=top_5,
                ebird_info=ebird_info,
                taxonomy=taxonomy,
                bird_id=bird_id,
                sensor=sensor,
            )

            common_name = taxonomy.get("common_name", predicted_species.replace("_", " ")) if taxonomy else predicted_species.replace("_", " ")
            scientific_name = taxonomy.get("scientific_name", "") if taxonomy else ""
            conservation_status = taxonomy.get("conservation_status", "") if taxonomy else ""
        except Exception as e:
            print(f"Supabase/eBird error: {e} — saving locally instead")
            save_pending(predicted_species, confidence, top_5, sensor=sensor)
            common_name = predicted_species.replace("_", " ")
            scientific_name = ""
            conservation_status = ""
    else:
        save_pending(predicted_species, confidence, top_5, sensor=sensor)
        common_name = predicted_species.replace("_", " ")
        scientific_name = ""
        conservation_status = ""

    send_detection(
        common_name=common_name,
        scientific_name=scientific_name,
        conservation_status=conservation_status,
        confidence=confidence,
        x=x, y=y, w=w, h=h,
    )
