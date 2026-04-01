"""Generate gallery data JSON with breed info for the image gallery view."""
import json
import logging
import random
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path("data/petfinder/train_images")
OUT_DIR = Path("ui/assets/samples")
DATA_DIR = Path("ui/assets/data")


def run_gallery_export():
    logger.info("Building gallery data with breed info...")

    df = pd.read_csv("data/petfinder/train/train.csv")
    breeds = pd.read_csv("data/petfinder/BreedLabels.csv")
    breed_map = dict(zip(breeds["BreedID"], breeds["BreedName"]))

    df["BreedName"] = df["Breed1"].map(breed_map).fillna("Mixed Breed")
    df["TypeName"] = df["Type"].map({1: "Dog", 2: "Cat"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gallery = {"Dog": {}, "Cat": {}}
    for type_name in ["Dog", "Cat"]:
        type_df = df[df["TypeName"] == type_name]
        top_breeds = type_df["BreedName"].value_counts().head(15).index.tolist()

        for breed in top_breeds:
            breed_df = type_df[type_df["BreedName"] == breed]
            breed_pets = breed_df.sample(n=min(100, len(breed_df)), random_state=42)
            breed_entries = []

            for _, row in breed_pets.iterrows():
                pet_id = row["PetID"]
                img_name = f"{pet_id}-1.jpg"
                src_path = SRC_DIR / img_name
                if not src_path.exists():
                    continue

                dst_name = f"gallery_{pet_id}.jpg"
                dst_path = OUT_DIR / dst_name

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)

                breed_entries.append({
                    "pet_id": pet_id,
                    "path": f"assets/samples/{dst_name}",
                    "speed": int(row["AdoptionSpeed"]),
                    "breed": breed,
                    "type": type_name,
                    "age": int(row["Age"]),
                    "name": str(row["Name"]) if pd.notna(row["Name"]) else "",
                })

            if breed_entries:
                gallery[type_name][breed] = breed_entries
                logger.info("  %s > %s: %d images", type_name, breed, len(breed_entries))

    out_path = DATA_DIR / "image_gallery.json"
    with open(out_path, "w") as f:
        json.dump(gallery, f)

    logger.info("Saved %s", out_path)

    breed_counts = {t: {b: len(v) for b, v in breeds_dict.items()} for t, breeds_dict in gallery.items()}
    logger.info("Gallery summary: %s", json.dumps(breed_counts, indent=2))


if __name__ == "__main__":
    run_gallery_export()
