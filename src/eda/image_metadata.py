import os
import glob
import json
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_metadata")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "samples")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, filename):
    with open(os.path.join(OUT_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
    logger.info(f"Saved {filename}")


def run_metadata_eda():
    logger.info("Extracting image metadata...")
    df = pd.read_csv(TRAIN_CSV)
    type_map = {1: 'Dog', 2: 'Cat'}
    pet_type = dict(zip(df['PetID'], df['Type'].map(type_map)))

    all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    total_images = len(all_images)
    logger.info(f"Total images found: {total_images}")

    # 1 — Overview
    avg_photos = float(df['PhotoAmt'].mean())
    median_photos = float(df['PhotoAmt'].median())

    np.random.seed(42)
    sample_paths = np.random.choice(all_images, min(3000, total_images), replace=False)

    widths, heights, file_sizes, types_list = [], [], [], []
    for p in sample_paths:
        try:
            fsize = os.path.getsize(p) / 1024  # KB
            with Image.open(p) as img:
                w, h = img.size
            pid = os.path.basename(p).rsplit('-', 1)[0]
            t = pet_type.get(pid, 'Unknown')
            widths.append(w)
            heights.append(h)
            file_sizes.append(round(fsize, 1))
            types_list.append(t)
        except Exception:
            pass

    median_w = int(np.median(widths)) if widths else 0
    median_h = int(np.median(heights)) if heights else 0
    avg_fsize_kb = float(np.mean(file_sizes)) if file_sizes else 0
    total_size_gb = round(avg_fsize_kb * total_images / 1024 / 1024, 2)

    save_json({
        'total_images': total_images,
        'avg_photos_per_pet': round(avg_photos, 2),
        'median_photos_per_pet': median_photos,
        'median_width': median_w,
        'median_height': median_h,
        'median_resolution': f"{median_w}×{median_h}",
        'total_size_gb': total_size_gb,
        'avg_file_size_kb': round(avg_fsize_kb, 1)
    }, 'image_overview.json')

    # 2 — Dimensions scatter + file sizes
    aspect_ratios = [round(w / h, 4) if h > 0 else 1.0 for w, h in zip(widths, heights)]
    save_json({
        'widths': widths,
        'heights': heights,
        'file_sizes': file_sizes,
        'types': types_list,
        'aspect_ratios': aspect_ratios,
        'reference_lines': [
            {'ratio': 0.75, 'label': '3:4'},
            {'ratio': 1.0, 'label': '1:1'},
            {'ratio': 1.33, 'label': '4:3'},
            {'ratio': 1.5, 'label': '3:2'},
            {'ratio': 2.0, 'label': '2:1'}
        ]
    }, 'image_dimensions.json')

    # 3 — Photo count analysis
    photo_amts = df['PhotoAmt'].tolist()
    speed_photo_stats = {}
    for speed in range(5):
        sub = df[df['AdoptionSpeed'] == speed]['PhotoAmt']
        speed_photo_stats[str(speed)] = {
            'values': sub.tolist(),
            'mean': round(float(sub.mean()), 2),
            'median': float(sub.median()),
            'q1': float(sub.quantile(0.25)),
            'q3': float(sub.quantile(0.75)),
        }

    fast_avg = round(float(df[df['AdoptionSpeed'] <= 1]['PhotoAmt'].mean()), 2)
    slow_avg = round(float(df[df['AdoptionSpeed'] >= 3]['PhotoAmt'].mean()), 2)
    corr_val = round(float(df['PhotoAmt'].corr(df['AdoptionSpeed'])), 4)

    save_json({
        'photo_amounts': photo_amts,
        'speed_stats': speed_photo_stats,
        'fast_adopted_avg': fast_avg,
        'slow_adopted_avg': slow_avg,
        'correlation': corr_val
    }, 'image_photo_count.json')

    # 4 — Copy sample grid images (5 per speed)
    logger.info("Copying sample grid images...")
    sample_grid = {}
    for speed in range(5):
        pet_ids = df[df['AdoptionSpeed'] == speed].sample(
            n=min(5, (df['AdoptionSpeed'] == speed).sum()), random_state=42
        )['PetID'].tolist()
        samples = []
        for pid in pet_ids:
            src = os.path.join(IMG_DIR, f"{pid}-1.jpg")
            if os.path.exists(src):
                dst_name = f"grid_{speed}_{pid}.jpg"
                shutil.copy2(src, os.path.join(SAMPLES_DIR, dst_name))
                samples.append({
                    'pet_id': pid,
                    'path': f"assets/samples/{dst_name}",
                    'speed': speed
                })
        sample_grid[str(speed)] = samples

    save_json({'grid': sample_grid}, 'image_samples.json')
    logger.info("Image metadata extraction complete.")


if __name__ == "__main__":
    run_metadata_eda()
