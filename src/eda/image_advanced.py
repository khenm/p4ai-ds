import os
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_advanced")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
BREED_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "BreedLabels.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data")
QUALITY_CSV = os.path.join(OUT_DIR, "image_quality_raw.csv")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ['Brightness', 'Blurriness', 'Contrast', 'Colorfulness', 'Saturation']


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


def run_breed_cluster_eda(min_samples: int = 10, n_clusters: int = 5, top_n: int = 20):
    """Compute per-breed mean image features, pairwise similarity, and cluster groups."""
    logger.info("Running breed image cluster analysis...")

    if not os.path.exists(QUALITY_CSV):
        logger.error("image_quality_raw.csv not found. Run image_quality.py first.")
        return

    qdf = pd.read_csv(QUALITY_CSV)
    df = pd.read_csv(TRAIN_CSV)
    breeds_df = pd.read_csv(BREED_CSV)
    breed_map = dict(zip(breeds_df['BreedID'], breeds_df['BreedName']))
    df['BreedName'] = df['Breed1'].map(breed_map).fillna('Mixed Breed')
    df['TypeName'] = df['Type'].map({1: 'Dog', 2: 'Cat'})

    # Merge quality data with breed info
    merged = qdf.merge(df[['PetID', 'BreedName', 'TypeName']], on='PetID', how='inner')

    # Standard-scale features globally so comparisons are on the same scale
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(merged[METRICS])
    merged = merged.copy()
    merged[METRICS] = feat_scaled

    result = {}
    for type_name in ['Dog', 'Cat']:
        sub = merged[merged['TypeName'] == type_name]

        # Keep only breeds with enough samples
        counts = sub['BreedName'].value_counts()
        top_breeds = counts[counts >= min_samples].head(top_n).index.tolist()
        if len(top_breeds) < 3:
            logger.warning(f"Not enough breeds for {type_name} — skipping")
            continue

        sub = sub[sub['BreedName'].isin(top_breeds)]

        # Mean feature vector per breed (rows = breeds, cols = features)
        means = sub.groupby('BreedName')[METRICS].mean().reindex(top_breeds)
        X = means.values  # shape (n_breeds, 5)

        # ── Pairwise cosine similarity ─────────────────────────────────────
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        x_norm = X / np.clip(norms, 1e-9, None)
        similarity = (x_norm @ x_norm.T).clip(-1, 1)

        # ── Agglomerative clustering on mean vectors ───────────────────────
        n_cl = min(n_clusters, len(top_breeds))
        clustering = AgglomerativeClustering(n_clusters=n_cl, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(X).tolist()

        # Map each breed to its cluster label
        breed_cluster = dict(zip(top_breeds, cluster_labels))

        # Cluster composition
        cluster_breeds = {str(c): [b for b in top_breeds if breed_cluster[b] == c]
                          for c in range(n_cl)}

        # ── Cluster vs Adoption Speed (from full un-scaled df) ─────────────
        sub_full = merged[merged['TypeName'] == type_name].copy()
        sub_full = sub_full[sub_full['BreedName'].isin(top_breeds)].copy()
        # Restore original AdoptionSpeed from qdf
        sub_full = qdf.merge(
            df[['PetID', 'BreedName', 'TypeName']], on='PetID', how='inner'
        ).query(f"TypeName == '{type_name}' and BreedName in @top_breeds").copy()
        sub_full['Cluster'] = sub_full['BreedName'].map(breed_cluster)

        ct = pd.crosstab(sub_full['Cluster'], sub_full['AdoptionSpeed'], normalize='index')
        cluster_speed = {}
        for c in range(n_cl):
            cluster_speed[str(c)] = [
                round(float(ct.loc[c, s]) if (c in ct.index and s in ct.columns) else 0, 4)
                for s in range(5)
            ]

        # ── Breed vs Adoption Speed (image-sampled) ────────────────────────
        ct_b = pd.crosstab(sub_full['BreedName'], sub_full['AdoptionSpeed'], normalize='index')
        ct_b = ct_b.reindex(top_breeds).fillna(0)
        breed_speed = {
            str(s): [round(float(ct_b.loc[b, s]) if (b in ct_b.index and s in ct_b.columns) else 0, 4)
                     for b in top_breeds]
            for s in range(5)
        }

        # Mean feature profiles per cluster (for radar/bar interpretation)
        cluster_profiles = {}
        for c in range(n_cl):
            c_breeds = cluster_breeds[str(c)]
            c_rows = means.loc[c_breeds]
            cluster_profiles[str(c)] = {
                feat: round(float(c_rows[feat].mean()), 4) for feat in METRICS
            }

        result[type_name] = {
            'breeds': top_breeds,
            'cluster_labels': cluster_labels,
            'n_clusters': n_cl,
            'cluster_breeds': cluster_breeds,
            'cluster_speed': cluster_speed,
            'breed_speed': breed_speed,
            'similarity': [[round(float(v), 4) for v in row] for row in similarity],
            'feature_names': METRICS,
            'cluster_profiles': cluster_profiles,
        }
        logger.info(f"  {type_name}: {len(top_breeds)} breeds → {n_cl} clusters")

    # ── Cross-correlation: Dog breeds × Cat breeds ────────────────────────
    if 'Dog' in result and 'Cat' in result:
        dog_breeds = result['Dog']['breeds']
        cat_breeds = result['Cat']['breeds']

        dog_sub = merged[merged['TypeName'] == 'Dog']
        cat_sub = merged[merged['TypeName'] == 'Cat']
        dog_means = dog_sub[dog_sub['BreedName'].isin(dog_breeds)].groupby('BreedName')[METRICS].mean().reindex(dog_breeds)
        cat_means = cat_sub[cat_sub['BreedName'].isin(cat_breeds)].groupby('BreedName')[METRICS].mean().reindex(cat_breeds)

        d_norms = np.linalg.norm(dog_means.values, axis=1, keepdims=True)
        c_norms = np.linalg.norm(cat_means.values, axis=1, keepdims=True)
        d_norm = dog_means.values / np.clip(d_norms, 1e-9, None)
        c_norm = cat_means.values / np.clip(c_norms, 1e-9, None)
        cross_sim = (d_norm @ c_norm.T).clip(-1, 1)  # (n_dog, n_cat)

        result['cross_similarity'] = {
            'dog_breeds': dog_breeds,
            'cat_breeds': cat_breeds,
            'similarity': [[round(float(v), 4) for v in row] for row in cross_sim],
        }
        logger.info(f"  Cross-similarity: {len(dog_breeds)} dogs × {len(cat_breeds)} cats")

    # ── Combined clustering: all breeds together ──────────────────────────
    all_breeds, all_types, all_means_rows = [], [], []
    for type_name in ['Dog', 'Cat']:
        if type_name not in result:
            continue
        for breed in result[type_name]['breeds']:
            sub_t = merged[merged['TypeName'] == type_name]
            row = sub_t[sub_t['BreedName'] == breed][METRICS].mean().values
            all_breeds.append(breed)
            all_types.append(type_name)
            all_means_rows.append(row)

    if len(all_breeds) >= 4:
        X_all = np.array(all_means_rows)
        n_cl_all = min(6, len(all_breeds))
        comb_clustering = AgglomerativeClustering(n_clusters=n_cl_all, metric='euclidean', linkage='ward')
        comb_labels = comb_clustering.fit_predict(X_all).tolist()

        breed_cluster_map = dict(zip(all_breeds, comb_labels))

        # Cluster composition with type info
        comb_cluster_breeds = {}
        for c in range(n_cl_all):
            comb_cluster_breeds[str(c)] = [
                {'breed': b, 'type': t}
                for b, t, cl in zip(all_breeds, all_types, comb_labels) if cl == c
            ]

        # Cluster vs adoption speed (combined)
        full_merged = qdf.merge(df[['PetID', 'BreedName', 'TypeName']], on='PetID', how='inner')
        full_merged = full_merged[full_merged['BreedName'].isin(all_breeds)].copy()
        full_merged['Cluster'] = full_merged['BreedName'].map(breed_cluster_map)
        ct_comb = pd.crosstab(full_merged['Cluster'], full_merged['AdoptionSpeed'], normalize='index')
        comb_cluster_speed = {}
        for c in range(n_cl_all):
            comb_cluster_speed[str(c)] = [
                round(float(ct_comb.loc[c, s]) if (c in ct_comb.index and s in ct_comb.columns) else 0, 4)
                for s in range(5)
            ]

        result['combined'] = {
            'breeds': all_breeds,
            'types': all_types,
            'cluster_labels': comb_labels,
            'n_clusters': n_cl_all,
            'cluster_breeds': comb_cluster_breeds,
            'cluster_speed': comb_cluster_speed,
        }
        logger.info(f"  Combined: {len(all_breeds)} breeds → {n_cl_all} clusters")

    save_json(result, 'image_breed_clusters.json')
    logger.info("Breed cluster analysis complete.")


if __name__ == "__main__":
    run_advanced_eda()
