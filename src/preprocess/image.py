import os
import json
import shutil
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_vertices(vertices):
    """Parse bounding box vertices to [xmin, ymin, xmax, ymax]."""
    x_coords = [v.get("x", 0) for v in vertices]
    y_coords = [v.get("y", 0) for v in vertices]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def extract_metadata(metadata_path, top_k):
    """Extract metadata from JSON file."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    labels = []
    if "labelAnnotations" in data:
        sorted_labels = sorted(data["labelAnnotations"], key=lambda x: x.get("score", 0), reverse=True)
        for label in sorted_labels[:top_k]:
            labels.append({
                "description": label.get("description", ""),
                "score": label.get("score", 0)
            })

    colors = []
    if "imagePropertiesAnnotation" in data:
        dom_colors = data["imagePropertiesAnnotation"].get("dominantColors", {}).get("colors", [])
        for c in dom_colors:
            color_info = c.get("color", {})
            colors.append({
                "red": color_info.get("red", 0),
                "green": color_info.get("green", 0),
                "blue": color_info.get("blue", 0),
                "score": c.get("score", 0)
            })

    bboxes = []
    if "cropHintsAnnotation" in data:
        hints = data["cropHintsAnnotation"].get("cropHints", [])
        for hint in hints:
            vertices = hint.get("boundingPoly", {}).get("vertices", [])
            if vertices:
                bboxes.append(parse_vertices(vertices))

    return {
        "labels": labels,
        "colors": colors,
        "bboxes": bboxes
    }


def process_split(split_name, raw_dir, images_dir, top_k):
    """Process a single data split."""
    split_images_dir = raw_dir / f"{split_name}_images"
    split_metadata_dir = raw_dir / f"{split_name}_metadata"

    target_split_dir = images_dir / split_name
    target_split_dir.mkdir(parents=True, exist_ok=True)

    coco_data = {
        "images": [],
        "categories": [{"id": 1, "name": "pet"}],
        "annotations": []
    }

    if not split_images_dir.exists():
        logger.warning(f"Images directory {split_images_dir} not found. Skipping {split_name}.")
        return

    image_files = list(split_images_dir.glob("*.jpg"))
    annotation_id = 1

    for idx, img_path in enumerate(tqdm(image_files, desc=f"Processing {split_name}")):
        target_img_path = target_split_dir / img_path.name
        shutil.copy2(img_path, target_img_path)

        # Get dim if needed
        try:
            with Image.open(target_img_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"Could not read dimensions for {img_path}: {e}")
            width, height = 0, 0

        image_id = idx + 1
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        meta_path = split_metadata_dir / f"{img_path.stem}.json"
        if meta_path.exists():
            meta = extract_metadata(meta_path, top_k)
            for bbox in meta["bboxes"]:
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin
                if w > 0 and h > 0:
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [xmin, ymin, w, h],
                        "labels": meta["labels"],
                        "colors": meta["colors"],
                        "iscrowd": 0,
                        "area": w * h
                    })
                    annotation_id += 1

    out_json = images_dir / f"{split_name}_annotations.json"
    logger.info(f"Saving annotations to {out_json}")
    with open(out_json, "w") as f:
        json.dump(coco_data, f, indent=4)


