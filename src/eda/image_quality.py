import os
import json
import shutil
import cv2
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_quality")

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


def _colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return std_root + 0.3 * mean_root


def run_quality_eda(sample_size=3000):
    logger.info(f"Extracting quality metrics for N={sample_size}...")
    df = pd.read_csv(TRAIN_CSV)
    type_map = {1: 'Dog', 2: 'Cat'}

    np.random.seed(42)
    sample_ids = df['PetID'].sample(n=min(sample_size, len(df)), random_state=42).tolist()

    results = []
    for i, pid in enumerate(sample_ids):
        row = df[df['PetID'] == pid].iloc[0]
        speed = int(row['AdoptionSpeed'])
        ptype = type_map.get(int(row['Type']), 'Unknown')
        img_path = os.path.join(IMG_DIR, f"{pid}-1.jpg")

        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        # RGB means (for 3D color space scatter)
        b_mean, g_mean, r_mean = cv2.mean(img_resized)[:3]

        results.append({
            'PetID': pid,
            'AdoptionSpeed': speed,
            'Type': ptype,
            'Brightness': round(float(np.mean(gray)), 2),
            'Blurriness': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            'Contrast': round(float(np.std(gray)), 2),
            'Colorfulness': round(float(_colorfulness(img_resized)), 2),
            'Saturation': round(float(np.mean(hsv[:, :, 1])), 2),
            'R_mean': round(float(r_mean), 1),
            'G_mean': round(float(g_mean), 1),
            'B_mean': round(float(b_mean), 1),
        })

        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{sample_size} images...")

    qdf = pd.DataFrame(results)
    logger.info(f"Valid quality samples: {len(qdf)}")

    # Composite quality score (min-max scaled average)
    metrics = ['Brightness', 'Blurriness', 'Contrast', 'Colorfulness', 'Saturation']
    for m in metrics:
        mn, mx = qdf[m].min(), qdf[m].max()
        qdf[f'{m}_s'] = (qdf[m] - mn) / (mx - mn) if mx > mn else 0.0
    qdf['QualityScore'] = qdf[[f'{m}_s' for m in metrics]].mean(axis=1)

    # Save raw data for image_advanced.py
    qdf.to_csv(os.path.join(OUT_DIR, 'image_quality_raw.csv'), index=False)

    # 1 — Quality scatter (sharpness vs contrast by type)
    save_json({
        'sharpness': qdf['Blurriness'].tolist(),
        'contrast': qdf['Contrast'].tolist(),
        'types': qdf['Type'].tolist(),
        'brightness': qdf['Brightness'].tolist(),
    }, 'image_quality_scatter.json')

    # 2 — RGB color space 3D scatter
    save_json({
        'r': qdf['R_mean'].tolist(),
        'g': qdf['G_mean'].tolist(),
        'b': qdf['B_mean'].tolist(),
        'brightness': qdf['Brightness'].tolist(),
    }, 'image_color_space.json')

    # 3 — Quality metrics table (median by speed)
    table = {}
    for m in metrics:
        table[m] = {}
        for speed in range(5):
            sub = qdf[qdf['AdoptionSpeed'] == speed][m]
            table[m][str(speed)] = round(float(sub.median()), 2) if len(sub) > 0 else 0
    save_json({'metrics': metrics, 'table': table}, 'image_quality_table.json')

    # 4 — Composite score by speed (violin/box)
    composite_by_speed = {}
    for speed in range(5):
        composite_by_speed[str(speed)] = qdf[qdf['AdoptionSpeed'] == speed]['QualityScore'].round(4).tolist()
    save_json({
        'scores_by_speed': composite_by_speed,
        'overall_mean': round(float(qdf['QualityScore'].mean()), 4),
        'overall_std': round(float(qdf['QualityScore'].std()), 4),
    }, 'image_quality_composite.json')

    # 5 — Best & worst images
    top8 = qdf.nlargest(8, 'QualityScore')
    bot8 = qdf.nsmallest(8, 'QualityScore')

    def _copy_and_record(subset, prefix):
        records = []
        for _, row in subset.iterrows():
            pid = row['PetID']
            src = os.path.join(IMG_DIR, f"{pid}-1.jpg")
            if os.path.exists(src):
                dst = f"{prefix}_{pid}.jpg"
                shutil.copy2(src, os.path.join(SAMPLES_DIR, dst))
                records.append({
                    'pet_id': pid,
                    'path': f"assets/samples/{dst}",
                    'score': round(float(row['QualityScore']), 4),
                    'speed': int(row['AdoptionSpeed']),
                    'type': row['Type'],
                })
        return records

    save_json({
        'best': _copy_and_record(top8, 'best'),
        'worst': _copy_and_record(bot8, 'worst'),
    }, 'image_best_worst.json')

    logger.info("Image quality extraction complete.")


if __name__ == "__main__":
    run_quality_eda()
