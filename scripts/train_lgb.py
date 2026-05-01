import os
import logging
import sys
import argparse
import warnings
import torch
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore', message='X does not have valid feature names')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.petfinder import PetFinderDataset, get_transforms
from src.models.cnn import TwoStageResNet
from src.utils.env import load_config, seed_everything
from src.utils.reporting import build_adoption_speed_section, build_breed_section, save_report
from src.analysis.ablation import compute_feature_slices, run_ablation_combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM for PetFinder")
    parser.add_argument('--config', type=str, default='configs/train_lgbm.yaml', help='Path to config file')
    return parser.parse_args()

def extract_features(loader, model, device):
    model.eval()
    all_features = []
    all_targets = {
        'AdoptionSpeed': [],
        'Type': [],
        'FurLength': [],
        'MaturitySize': [],
        'Breed1': [],
        'Health': [],
        'Vaccinated': [],
        'Dewormed': [],
        'Sterilized': [],
        'Gender': [],
        'Color1': [],
        'PhotoAmt': []
    }
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            
            for k in all_targets.keys():
                all_targets[k].append(targets[k].numpy())
                
    return np.vstack(all_features), {k: np.concatenate(v) for k, v in all_targets.items()}

def main():
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.get('seed_value', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    lgb_device = 'gpu' if torch.cuda.is_available() else 'cpu'

    raw_dir = config['data']['raw_dir']
    csv_path = os.path.join(raw_dir, 'train', 'train.csv')
    img_dir = os.path.join(raw_dir, 'train_images')
    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)

    dataset = PetFinderDataset(csv_path, img_dir, transform=get_transforms(train=False))
    
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TwoStageResNet(extract_features=True).to(device)
    
    logger.info("Extracting features for train set...")
    X_train_img, y_train_dict = extract_features(train_loader, model, device)
    
    logger.info("Extracting features for val set...")
    X_val_img, y_val_dict = extract_features(val_loader, model, device)
    
    logger.info("\n--- Stage 1: Predicting Tabular Features ---")
    models_stage1 = {}
    preds_stage1_train = []
    preds_stage1_val = []
    
    for feature in ['Type', 'FurLength', 'MaturitySize', 'Breed1', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized', 'Gender', 'Color1']:
        logger.info(f"Training LightGBM for {feature}...")
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, device=lgb_device, verbose=-1)
        clf.fit(X_train_img, y_train_dict[feature])
        train_preds = clf.predict_proba(X_train_img)
        val_preds = clf.predict_proba(X_val_img)
        
        preds_stage1_train.append(train_preds)
        preds_stage1_val.append(val_preds)
        models_stage1[feature] = clf
        
        val_acc = accuracy_score(y_val_dict[feature], clf.predict(X_val_img))
        logger.info(f"Val Accuracy for {feature}: {val_acc:.4f}")

    logger.info("\n--- Stage 2: Predicting AdoptionSpeed ---")
    
    X_train_concat = np.hstack([X_train_img] + preds_stage1_train + [y_train_dict['PhotoAmt'].reshape(-1, 1)])
    X_val_concat = np.hstack([X_val_img] + preds_stage1_val + [y_val_dict['PhotoAmt'].reshape(-1, 1)])
    
    clf_final = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, device=lgb_device, verbose=-1)
    clf_final.fit(X_train_concat, y_train_dict['AdoptionSpeed'])
    
    y_pred_val = clf_final.predict(X_val_concat)
    
    acc = accuracy_score(y_val_dict['AdoptionSpeed'], y_pred_val)
    qwk = cohen_kappa_score(y_val_dict['AdoptionSpeed'], y_pred_val, weights='quadratic')
    
    logger.info(f"\nFinal Validation Results:")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")

    stage1_accuracies = {
        feat: round(accuracy_score(y_val_dict[feat], models_stage1[feat].predict(X_val_img)), 4)
        for feat in models_stage1
    }

    logger.info("Running combinatorial ablation (2^N subsets)...")
    feature_slices = compute_feature_slices(512, models_stage1)
    ablation_result = run_ablation_combinations(
        clf_final, X_val_concat, y_val_dict['AdoptionSpeed'], feature_slices,
        max_combo_size=3,
    )
    logger.info(f"Ablation complete: {ablation_result['n_combinations']} combinations evaluated.")
    logger.info(f"Worst combo (biggest QWK drop): {ablation_result['worst'][0]}")

    report = {
        "accuracy": round(acc, 4),
        "qwk": round(qwk, 4),
        "stage1_val_accuracy": stage1_accuracies,
        "adoption_speed": build_adoption_speed_section(
            y_val_dict['AdoptionSpeed'], y_pred_val
        ),
        "breed1": build_breed_section(
            y_val_dict['Breed1'], models_stage1['Breed1'].predict(X_val_img)
        ),
        "ablation_combinations": ablation_result,
    }
    out_path = save_report(args.config, report)
    logger.info(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
