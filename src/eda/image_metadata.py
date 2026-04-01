import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import logging
from theme import set_theme

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_metadata")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def run_metadata_eda():
    logger.info("Extracting image metadata (resolutions, counts)...")
    df = pd.read_csv(TRAIN_CSV)
    colors = set_theme()
    
    # 1. Image Counts per PetID (Histogram)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['PhotoAmt'], bins=30, kde=False, color=colors['qualitative'][2])
    plt.title('Distribution of Number of Photos per Pet')
    plt.xlabel('Number of Photos')
    plt.savefig(os.path.join(OUT_DIR, 'img_photo_count_dist.png'), bbox_inches='tight')
    plt.close()
    
    # [NEW] 1.5 Photo Count vs Adoption Speed
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='AdoptionSpeed', y='PhotoAmt', palette=colors['sequential'])
    plt.title('Photo Count vs Adoption Speed')
    plt.xlabel('Adoption Speed (0=Fastest, 4=Not Adopted)')
    plt.ylabel('Number of Photos')
    # Limit y-axis if extreme outliers exist to make standard ranges visible:
    plt.ylim(0, df['PhotoAmt'].quantile(0.99))
    plt.savefig(os.path.join(OUT_DIR, 'img_photo_count_vs_speed.png'), bbox_inches='tight')
    plt.close()

    # Collect metadata for a subsample of images to evaluate dimensions
    image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    sample_paths = np.random.choice(image_paths, min(2000, len(image_paths)), replace=False)
    
    widths, heights, aspect_ratios = [], [], []
    for path in sample_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception:
            pass
            
    # 2. Resolution distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(widths, bins=30, color=colors['qualitative'][1], label='Width')
    plt.title('Image Widths')
    plt.subplot(1, 2, 2)
    sns.histplot(heights, bins=30, color=colors['qualitative'][0], label='Height')
    plt.title('Image Heights')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'img_resolution_dist.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Aspect Ratio distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(aspect_ratios, bins=40, color=colors['qualitative'][3], kde=True)
    plt.axvline(1.0, color=colors['diverging'][0], linestyle='--', label='Square 1:1')
    plt.title('Aspect Ratio (Width/Height) Distribution')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'img_aspect_ratio_dist.png'), bbox_inches='tight')
    plt.close()

    # 4. Sample Grid mapping categorical Adoption Speed
    logger.info("Generating Adoption Speed image grids...")
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    fig.suptitle('Sample Images by Adoption Speed (Row 0=Fastest, Row 4=Slowest/None)', fontsize=16)
    
    for speed_val in range(5):
        samples = df[df['AdoptionSpeed'] == speed_val].sample(10, random_state=42)
        found = 0
        for _, row in samples.iterrows():
            pet_id = row['PetID']
            img_path = os.path.join(IMG_DIR, f"{pet_id}-1.jpg")
            if os.path.exists(img_path) and found < 4:
                try:
                    img = Image.open(img_path)
                    ax = axes[speed_val, found]
                    ax.imshow(img.resize((150, 150)))
                    ax.set_title(f"Speed: {speed_val}")
                    ax.axis('off')
                    found += 1
                except:
                    pass
            if found >= 4:
                break
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, 'img_sample_grid.png'), bbox_inches='tight')
    plt.close()
    
    logger.info("Basic metadata tier 1 generated.")

if __name__ == "__main__":
    run_metadata_eda()
