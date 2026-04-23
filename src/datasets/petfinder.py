import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PetFinderDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.df = pd.read_csv(csv_path)
        valid_indices = []
        for idx, row in self.df.iterrows():
            pet_id = row['PetID']
            img_path = os.path.join(img_dir, f"{pet_id}-1.jpg")
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Loaded PetFinderDataset with {len(self.df)} images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pet_id = row['PetID']
        
        img_path = os.path.join(self.img_dir, f"{pet_id}-1.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        adoption_speed = torch.tensor(row['AdoptionSpeed'], dtype=torch.long)
        pet_type = torch.tensor(row['Type'] - 1, dtype=torch.long)
        photo_amt = torch.tensor(row['PhotoAmt'], dtype=torch.float32)
        fur_length = torch.tensor(row['FurLength'], dtype=torch.long)
        
        maturity_size = torch.tensor(row['MaturitySize'], dtype=torch.long)
        breed1 = torch.tensor(row['Breed1'], dtype=torch.long)
        health = torch.tensor(row['Health'], dtype=torch.long)
        vaccinated = torch.tensor(row['Vaccinated'], dtype=torch.long)
        dewormed = torch.tensor(row['Dewormed'], dtype=torch.long)
        sterilized = torch.tensor(row['Sterilized'], dtype=torch.long)
        gender = torch.tensor(row['Gender'], dtype=torch.long)
        color1 = torch.tensor(row['Color1'], dtype=torch.long)

        targets = {
            'AdoptionSpeed': adoption_speed,
            'Type': pet_type,
            'FurLength': fur_length,
            'MaturitySize': maturity_size,
            'Breed1': breed1,
            'Health': health,
            'Vaccinated': vaccinated,
            'Dewormed': dewormed,
            'Sterilized': sterilized,
            'Gender': gender,
            'Color1': color1,
            'PhotoAmt': photo_amt,
            'PetID': pet_id
        }
        
        return image, targets

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
