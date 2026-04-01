import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import logging
from theme import set_theme

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_advanced")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def get_image_dominant_colors(path, k=3):
    img = cv2.imread(path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    pixels = img.reshape((-1, 3))
    # KMeans to find dominant colors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(pixels)
    # Sort by weight/density
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    centers = kmeans.cluster_centers_
    sorted_centers = centers[np.argsort(-counts)]
    return sorted_centers / 255.0  # Normalize to 0-1 for matplotlib

def compute_dominant_colors_swatches(df, N=15):
    logger.info(f"Extracting dominant colors for N={N} per adoption speed...")
    colors = set_theme()
    fig, axes = plt.subplots(5, N, figsize=(15, 6))
    fig.suptitle('Dominant Color Palettes by Adoption Speed (Top 3 Colors per Image)', fontsize=16)
    
    for speed in range(5):
        sample_df = df[df['AdoptionSpeed'] == speed].sample(n=min(N, (df['AdoptionSpeed'] == speed).sum()), random_state=42)
        
        found = 0
        for _, row in sample_df.iterrows():
            pet_id = row['PetID']
            img_path = os.path.join(IMG_DIR, f"{pet_id}-1.jpg")
            
            ax = axes[speed, found]
            ax.axis('off')
            if os.path.exists(img_path):
                colors = get_image_dominant_colors(img_path, k=3)
                if len(colors) == 3:
                     for j, color in enumerate(colors):
                         rect = patches.Rectangle((j/3, 0), 1/3, 1, color=color)
                         ax.add_patch(rect)
            
            if found == 0:
                ax.set_title(f"Speed: {speed}", loc='left')
                
            found += 1
            if found >= N:
                break
                
        # Fill any empty slots
        while found < N:
            axes[speed, found].axis('off')
            found += 1
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, 'img_dominant_colors.png'), bbox_inches='tight')
    plt.close()

def run_advanced_eda():
    df = pd.read_csv(TRAIN_CSV)
    compute_dominant_colors_swatches(df, N=15)
    
    logger.info("Advanced tier 3 images generated.")

if __name__ == "__main__":
    run_advanced_eda()
