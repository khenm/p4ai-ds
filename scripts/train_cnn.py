import os
import logging
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.petfinder import PetFinderDataset, get_transforms
from src.models.cnn import TwoStageResNet
from src.utils.env import load_config, seed_everything

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Stage CNN for PetFinder")
    parser.add_argument('--config', type=str, default='configs/train_cnn.yaml', help='Path to config file')
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
    model = TwoStageResNet(extract_features=False).to(device)
    
    # 3. Optim & Loss
    lr = float(config.get('learning_rate', 3e-4))
    optimizer_stage1 = AdamW(model.parameters(), lr=lr)
    optimizer_stage2 = AdamW(model.head_adoption_speed.parameters(), lr=lr)
    
    criterion_adoption = nn.CrossEntropyLoss()
    criterion_type = nn.CrossEntropyLoss()
    criterion_fur = nn.CrossEntropyLoss()
    criterion_maturity = nn.CrossEntropyLoss()
    criterion_breed1 = nn.CrossEntropyLoss()
    criterion_health = nn.CrossEntropyLoss()
    criterion_vaccinated = nn.CrossEntropyLoss()
    criterion_dewormed = nn.CrossEntropyLoss()
    criterion_sterilized = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_color1 = nn.CrossEntropyLoss()

    epochs = config.get('max_epochs', 3)
    
    # ------------------ STAGE 1: Train Tabular Heads ------------------
    logger.info("=== STAGE 1: Training Tabular Heads ===")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in pbar:
            images = images.to(device)
            
            target_type = targets['Type'].to(device)
            target_fur = targets['FurLength'].to(device)
            target_maturity = targets['MaturitySize'].to(device)
            target_breed1 = targets['Breed1'].to(device)
            target_health = targets['Health'].to(device)
            target_vaccinated = targets['Vaccinated'].to(device)
            target_dewormed = targets['Dewormed'].to(device)
            target_sterilized = targets['Sterilized'].to(device)
            target_gender = targets['Gender'].to(device)
            target_color1 = targets['Color1'].to(device)
            
            optimizer_stage1.zero_grad()
            outputs = model(images)
            
            loss_type = criterion_type(outputs['Type'], target_type)
            loss_fur = criterion_fur(outputs['FurLength'], target_fur)
            loss_maturity = criterion_maturity(outputs['MaturitySize'], target_maturity)
            loss_breed1 = criterion_breed1(outputs['Breed1'], target_breed1)
            loss_health = criterion_health(outputs['Health'], target_health)
            loss_vaccinated = criterion_vaccinated(outputs['Vaccinated'], target_vaccinated)
            loss_dewormed = criterion_dewormed(outputs['Dewormed'], target_dewormed)
            loss_sterilized = criterion_sterilized(outputs['Sterilized'], target_sterilized)
            loss_gender = criterion_gender(outputs['Gender'], target_gender)
            loss_color1 = criterion_color1(outputs['Color1'], target_color1)
            
            loss = loss_type + loss_fur + loss_maturity + loss_breed1 + loss_health + loss_vaccinated + loss_dewormed + loss_sterilized + loss_gender + loss_color1
            
            loss.backward()
            optimizer_stage1.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    # ------------------ STAGE 2: Train AdoptionSpeed Head ------------------
    logger.info("=== STAGE 2: Training AdoptionSpeed Head (Frozen Stage 1) ===")
    model.freeze_stage1()
    
    for epoch in range(epochs):
        model.train() # The frozen parts will not update because requires_grad=False
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in pbar:
            images = images.to(device)
            target_adoption = targets['AdoptionSpeed'].to(device)
            
            optimizer_stage2.zero_grad()
            outputs = model(images)
            
            loss = criterion_adoption(outputs['AdoptionSpeed'], target_adoption)
            loss.backward()
            optimizer_stage2.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        correct_adoption = 0
        total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Stage 2 - Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                target_adoption = targets['AdoptionSpeed'].to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs['AdoptionSpeed'], 1)
                total += target_adoption.size(0)
                correct_adoption += (predicted == target_adoption).sum().item()
                
        val_acc = 100 * correct_adoption / total
        logger.info(f"Epoch {epoch+1} - Val AdoptionSpeed Accuracy: {val_acc:.2f}%\n")

if __name__ == "__main__":
    main()
