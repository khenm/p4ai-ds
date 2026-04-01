import os
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_advanced")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data")
QUALITY_CSV = os.path.join(OUT_DIR, "image_quality_raw.csv")
os.makedirs(OUT_DIR, exist_ok=True)


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


def _extract_dominant_colors(path, k=3):
    img = cv2.imread(path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    pixels = img.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(pixels)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_centers = kmeans.cluster_centers_[np.argsort(-counts)]
    return [[int(c) for c in color] for color in sorted_centers]


def run_advanced_eda():
    logger.info("Running advanced image analysis...")
    df = pd.read_csv(TRAIN_CSV)
    type_map = {1: 'Dog', 2: 'Cat'}

    # Load quality metrics from previous step
    if not os.path.exists(QUALITY_CSV):
        logger.error("image_quality_raw.csv not found. Run image_quality.py first.")
        return
    qdf = pd.read_csv(QUALITY_CSV)

    metrics = ['Brightness', 'Blurriness', 'Contrast', 'Colorfulness', 'Saturation']
    X = qdf[metrics].values

    # 1 — PCA
    logger.info("Running PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=len(metrics))
    pca.fit(X_scaled)
    save_json({
        'explained_variance': [round(float(v), 4) for v in pca.explained_variance_ratio_],
        'cumulative_variance': [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)],
        'components': metrics,
    }, 'image_pca.json')

    # 2 — t-SNE
    logger.info("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    save_json({
        'x': [round(float(v), 4) for v in X_tsne[:, 0]],
        'y': [round(float(v), 4) for v in X_tsne[:, 1]],
        'speeds': qdf['AdoptionSpeed'].tolist(),
        'types': qdf['Type'].tolist(),
    }, 'image_tsne.json')

    # 3 — Dominant colors by speed and type
    logger.info("Extracting dominant colors...")
    n_per_group = 20
    color_data = {'by_speed': {}, 'by_type': {}}

    for speed in range(5):
        pids = df[df['AdoptionSpeed'] == speed].sample(
            n=min(n_per_group, (df['AdoptionSpeed'] == speed).sum()), random_state=42
        )['PetID'].tolist()
        palettes = []
        for pid in pids:
            path = os.path.join(IMG_DIR, f"{pid}-1.jpg")
            if os.path.exists(path):
                colors = _extract_dominant_colors(path, k=3)
                if colors:
                    palettes.append({'pet_id': pid, 'colors': colors})
        color_data['by_speed'][str(speed)] = palettes

    for type_val, type_name in type_map.items():
        pids = df[df['Type'] == type_val].sample(
            n=min(n_per_group, (df['Type'] == type_val).sum()), random_state=42
        )['PetID'].tolist()
        palettes = []
        for pid in pids:
            path = os.path.join(IMG_DIR, f"{pid}-1.jpg")
            if os.path.exists(path):
                colors = _extract_dominant_colors(path, k=3)
                if colors:
                    palettes.append({'pet_id': pid, 'colors': colors})
        color_data['by_type'][type_name] = palettes

    save_json(color_data, 'image_dominant_colors.json')

    # 4 — Cross-modality: photo count × quality score interaction
    logger.info("Computing cross-modality analysis...")
    # Merge photo count with quality scores
    qdf_merged = qdf.merge(df[['PetID', 'PhotoAmt']], on='PetID', how='left')

    # Recompute quality score if not present
    for m in metrics:
        mn, mx = qdf_merged[m].min(), qdf_merged[m].max()
        qdf_merged[f'{m}_s'] = (qdf_merged[m] - mn) / (mx - mn) if mx > mn else 0.0
    qdf_merged['QualityScore'] = qdf_merged[[f'{m}_s' for m in metrics]].mean(axis=1)

    # Bin photo count and quality score
    photo_bins = [0, 1, 2, 3, 5, 10, 30]
    quality_bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    qdf_merged['PhotoBin'] = pd.cut(qdf_merged['PhotoAmt'], bins=photo_bins, right=True,
                                     labels=['1', '2', '3', '4-5', '6-10', '10+'])
    qdf_merged['QualityBin'] = pd.cut(qdf_merged['QualityScore'], bins=quality_bins, right=True,
                                       labels=['Very Low', 'Low', 'Medium', 'Good', 'High', 'Very High'])

    heatmap = qdf_merged.groupby(['PhotoBin', 'QualityBin'], observed=True)['AdoptionSpeed'].mean()
    heatmap_df = heatmap.unstack(fill_value=np.nan)

    # Quality by type
    type_quality = {}
    for t in ['Dog', 'Cat']:
        sub = qdf_merged[qdf_merged['Type'] == t]
        type_quality[t] = {m: round(float(sub[m].median()), 2) for m in metrics}
        type_quality[t]['QualityScore'] = round(float(sub['QualityScore'].median()), 4)

    save_json({
        'heatmap': {
            'x_labels': heatmap_df.columns.tolist(),
            'y_labels': heatmap_df.index.tolist(),
            'values': [[round(float(v), 2) if not np.isnan(v) else None for v in row]
                        for row in heatmap_df.values],
        },
        'type_quality': type_quality,
    }, 'image_cross_modality.json')

    logger.info("Advanced image analysis complete.")


if __name__ == "__main__":
    run_advanced_eda()
