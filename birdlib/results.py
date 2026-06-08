from birdlib.ebird import get_species_info
from birdlib.supabase import insert_sighting

def print_results(predicted_species, confidence, top_5, top_n):
    print("\n--- Inference Result ---")
    print(f"Top {top_n} predictions:")

    for species, score in top_5.items():
        print(f"{species}: {score:.4f}")

    print(f"Predicted class: {predicted_species} with confidence {confidence:.4f}")

def send_results(predicted_species, confidence, top_5):

    if predicted_species == "unknown":
        print("Unknown species — skipping eBird and Supabase")
        return

    ebird_info = get_species_info(predicted_species)

    insert_sighting(
        predicted_species=predicted_species,
        confidence=confidence,
        top_5=top_5,
        ebird_info=ebird_info
    )