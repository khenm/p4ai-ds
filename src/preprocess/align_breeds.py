"""Align breed labels from CSV to COCO-style annotation JSON files.

Reads BreedLabels.csv and train/test CSVs to inject breed metadata into
train_annotations.json and test_annotations.json.
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PETFINDER_DIR = PROJECT_ROOT / "data" / "petfinder"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

BREED_LABELS_PATH = PETFINDER_DIR / "BreedLabels.csv"
TRAIN_CSV_PATH = PETFINDER_DIR / "train" / "train.csv"
TEST_CSV_PATH = PETFINDER_DIR / "test" / "test.csv"

TRAIN_ANNOTATIONS_PATH = IMAGES_DIR / "train_annotations.json"
TEST_ANNOTATIONS_PATH = IMAGES_DIR / "test_annotations.json"

TYPE_MAP = {1: "Dog", 2: "Cat"}


def load_breed_labels(path: Path) -> dict[int, str]:
    """Build BreedID -> BreedName lookup from BreedLabels.csv."""
    breed_map: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            breed_id = int(row["BreedID"])
            breed_map[breed_id] = row["BreedName"].strip('"')
    breed_map[0] = "Not Specified"
    logger.info("Loaded %d breed labels", len(breed_map))
    return breed_map


def load_pet_breeds(csv_path: Path) -> dict[str, dict]:
    """Build PetID -> {type, breed1_id, breed2_id} lookup from train/test CSV."""
    pet_map: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pet_id = row["PetID"]
            pet_map[pet_id] = {
                "type": int(row["Type"]),
                "breed1_id": int(row["Breed1"]),
                "breed2_id": int(row["Breed2"]),
            }
    logger.info("Loaded %d pets from %s", len(pet_map), csv_path.name)
    return pet_map


def align_annotations(
    annotations_path: Path,
    pet_breeds: dict[str, dict],
    breed_labels: dict[int, str],
) -> None:
    """Add breed information to each annotation and update categories."""
    with open(annotations_path, encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_pet: dict[int, dict] = {}
    for img in coco["images"]:
        pet_id = img["file_name"].rsplit("-", 1)[0]
        if pet_id in pet_breeds:
            image_id_to_pet[img["id"]] = pet_breeds[pet_id]

    matched, unmatched = 0, 0
    breed_category_set: set[tuple[int, str]] = set()

    for ann in coco["annotations"]:
        pet_info = image_id_to_pet.get(ann["image_id"])
        if pet_info is None:
            unmatched += 1
            ann["breed1"] = None
            ann["breed2"] = None
            ann["breed1_name"] = "Unknown"
            ann["breed2_name"] = "Unknown"
            ann["pet_type"] = "Unknown"
            continue

        matched += 1
        b1 = pet_info["breed1_id"]
        b2 = pet_info["breed2_id"]
        b1_name = breed_labels.get(b1, f"Unknown({b1})")
        b2_name = breed_labels.get(b2, f"Unknown({b2})")

        ann["breed1"] = b1
        ann["breed2"] = b2
        ann["breed1_name"] = b1_name
        ann["breed2_name"] = b2_name
        ann["pet_type"] = TYPE_MAP.get(pet_info["type"], "Unknown")

        breed_category_set.add((b1, b1_name))
        if b2 != 0:
            breed_category_set.add((b2, b2_name))

    sorted_breeds = sorted(breed_category_set, key=lambda x: x[0])
    coco["breed_categories"] = [
        {"id": bid, "name": bname} for bid, bname in sorted_breeds
    ]

    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)

    logger.info(
        "Updated %s: %d matched, %d unmatched, %d unique breeds",
        annotations_path.name,
        matched,
        unmatched,
        len(breed_category_set),
    )


def main() -> None:
    breed_labels = load_breed_labels(BREED_LABELS_PATH)

    train_breeds = load_pet_breeds(TRAIN_CSV_PATH)
    test_breeds = load_pet_breeds(TEST_CSV_PATH)

    if TRAIN_ANNOTATIONS_PATH.exists():
        align_annotations(TRAIN_ANNOTATIONS_PATH, train_breeds, breed_labels)
    else:
        logger.warning("Train annotations not found at %s", TRAIN_ANNOTATIONS_PATH)

    if TEST_ANNOTATIONS_PATH.exists():
        align_annotations(TEST_ANNOTATIONS_PATH, test_breeds, breed_labels)
    else:
        logger.warning("Test annotations not found at %s", TEST_ANNOTATIONS_PATH)

    logger.info("Breed alignment complete")


if __name__ == "__main__":
    main()
