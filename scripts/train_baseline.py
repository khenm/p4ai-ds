import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.petfinder import PetFinderDataset, get_transforms
from src.models.cnn import BaselineResNet
from src.utils.env import load_config, seed_everything
from src.utils.reporting import build_clf_section, save_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN for PetFinder")
    parser.add_argument('--config', type=str, default='configs/train_baseline.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.get('seed_value', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Dataset & DataLoader
    raw_dir = config['data']['raw_dir']
    csv_path = os.path.join(raw_dir, 'train', 'train.csv')
    img_dir = os.path.join(raw_dir, 'train_images')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    full_dataset = PetFinderDataset(csv_path, img_dir, transform=get_transforms(train=True))
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2. Model
    model = BaselineResNet(num_classes=5).to(device)
    
    # 3. Optim & Loss
    lr = float(config.get('learning_rate', 3e-4))
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epochs = config.get('max_epochs', 10)

    # Training Loop
    logger.info("=== Training Baseline ResNet ===")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in pbar:
            images = images.to(device)
            target_adoption = targets['AdoptionSpeed'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, target_adoption)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                target_adoption = targets['AdoptionSpeed'].to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target_adoption.cpu().numpy())
                
        val_acc = 100 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        val_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
        logger.info(f"Epoch {epoch+1} - Val AdoptionSpeed — Acc: {val_acc:.2f}% | QWK: {val_qwk:.4f}\n")

    # --- Classification Report ---
    ADOPTION_CLASSES = ["Speed 0", "Speed 1", "Speed 2", "Speed 3", "Speed 4"]
    report = {
        "exp_name": config.get("exp_name", ""),
        **build_clf_section(all_targets, all_preds, ADOPTION_CLASSES),
        "accuracy": round(val_acc, 2),
        "qwk": round(val_qwk, 4),
    }
    report_path = save_report(args.config, report)
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
