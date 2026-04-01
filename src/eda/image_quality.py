import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from theme import set_theme
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("image_quality")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "petfinder", "train_images")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def calculate_colorfulness(image):
    # Hasler & Süsstrunk metric
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def extract_quality_metrics(df, sample_size=3000):
    logger.info(f"Extracting photo quality metrics for N={sample_size} photos...")
    
    np.random.seed(42)
    df_valid = df.copy()
    sample_pet_ids = df_valid['PetID'].sample(n=min(sample_size, len(df_valid)), random_state=42).tolist()
    
    results = []
    
    for i, pet_id in enumerate(sample_pet_ids):
        speed = df_valid.loc[df_valid['PetID'] == pet_id, 'AdoptionSpeed'].values[0]
        # Evaluate primary photo only
        img_path = os.path.join(IMG_DIR, f"{pet_id}-1.jpg")
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to standard size to normalize metrics speed and domain
                img = cv2.resize(img, (256, 256))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                brightness = np.mean(gray)
                blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
                contrast = np.std(gray)
                colorfulness = calculate_colorfulness(img)
                saturation = np.mean(hsv[:, :, 1])
                
                results.append({
                    'PetID': pet_id,
                    'AdoptionSpeed': speed,
                    'Brightness': brightness,
                    'Blurriness': blurriness,
                    'Contrast': contrast,
                    'Colorfulness': colorfulness,
                    'Saturation': saturation
                })
        
        if (i+1) % 500 == 0:
            logger.info(f"Processed {i+1} images...")
            
    return pd.DataFrame(results)

def plot_quality_distributions():
    df = pd.read_csv(TRAIN_CSV)
    q_df = extract_quality_metrics(df, sample_size=3000)
    
    if len(q_df) == 0:
        logger.error("No valid images found for quality extraction.")
        return
        
    metrics = ['Brightness', 'Blurriness', 'Contrast', 'Colorfulness', 'Saturation']
    colors = set_theme()
    
    # Print summary table to terminal
    logger.info("Median values of raw pixel statistics per Adoption Speed (showing no strong signal):")
    print(q_df.groupby('AdoptionSpeed')[metrics].median())

    # Calculate Composite Photo Quality Index
    # Min-max scale each metric
    for m in metrics:
        min_val = q_df[m].min()
        max_val = q_df[m].max()
        if max_val > min_val:
            q_df[f'{m}_scaled'] = (q_df[m] - min_val) / (max_val - min_val)
        else:
            q_df[f'{m}_scaled'] = 0.0

    scaled_metrics = [f'{m}_scaled' for m in metrics]
    q_df['QualityScore'] = q_df[scaled_metrics].mean(axis=1)

    # 1. Plot Composite Index
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=q_df, x='AdoptionSpeed', y='QualityScore', inner="quartile", palette=colors['sequential'])
    plt.title('Composite Photo Quality Index vs Adoption Speed')
    plt.ylabel('Quality Score (Normalized)')
    plt.savefig(os.path.join(OUT_DIR, 'img_quality_composite_score.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Best vs Worst Photos grid by Composite Score
    logger.info("Generating Best vs Worst Photos Grid based on Composite Quality Index...")
    top_5 = q_df.nlargest(5, 'QualityScore')
    bottom_5 = q_df.nsmallest(5, 'QualityScore')
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Highest vs Lowest Quality Photos (by Composite Index)', fontsize=16)
    
    # Plot top 5
    for idx, (_, row) in enumerate(top_5.iterrows()):
        pet_id = row['PetID']
        score = row['QualityScore']
        img_path = os.path.join(IMG_DIR, f"{pet_id}-1.jpg")
        ax = axes[0, idx]
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img.resize((150, 150)))
                ax.set_title(f"Best: {score:.3f}")
            except Exception:
                pass
        ax.axis('off')

    # Plot bottom 5
    for idx, (_, row) in enumerate(bottom_5.iterrows()):
        pet_id = row['PetID']
        score = row['QualityScore']
        img_path = os.path.join(IMG_DIR, f"{pet_id}-1.jpg")
        ax = axes[1, idx]
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img.resize((150, 150)))
                ax.set_title(f"Worst: {score:.3f}")
            except Exception:
                pass
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, 'img_best_worst_quality.png'), bbox_inches='tight')
    plt.close()

    logger.info("Quality distribution tier 2 generated.")

if __name__ == "__main__":
    plot_quality_distributions()
