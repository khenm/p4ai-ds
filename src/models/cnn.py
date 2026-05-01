import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaselineResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineResNet, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

class TwoStageResNet(nn.Module):
    def __init__(self, extract_features=False):
        super(TwoStageResNet, self).__init__()
        self.extract_features = extract_features
        
        # Stage 1: Backbone
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Stage 1: Tabular Heads
        if not extract_features:
            self.head_type = nn.Linear(self.num_ftrs, 2)
            self.head_furlength = nn.Linear(self.num_ftrs, 4)
            self.head_maturitysize = nn.Linear(self.num_ftrs, 5)
            self.head_breed1 = nn.Linear(self.num_ftrs, 308)
            self.head_health = nn.Linear(self.num_ftrs, 4)
            self.head_vaccinated = nn.Linear(self.num_ftrs, 4)
            self.head_dewormed = nn.Linear(self.num_ftrs, 4)
            self.head_sterilized = nn.Linear(self.num_ftrs, 4)
            self.head_gender = nn.Linear(self.num_ftrs, 4)
            self.head_color1 = nn.Linear(self.num_ftrs, 8)
            
            # Stage 2: Adoption Speed Head
            # Input dim = 512 (image) + sum of tabular logits dims (2+4+5+308+4+4+4+4+4+8) = 512 + 347 = 859
            tabular_dims = 2 + 4 + 5 + 308 + 4 + 4 + 4 + 4 + 4 + 8
            concat_dim = self.num_ftrs + tabular_dims
            
            self.head_adoption_speed = nn.Sequential(
                nn.Linear(concat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 5)
            )

    def freeze_stage1(self):
        """Freeze the backbone and tabular heads for Stage 2 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for head in [self.head_type, self.head_furlength, self.head_maturitysize, 
                     self.head_breed1, self.head_health, self.head_vaccinated,
                     self.head_dewormed, self.head_sterilized, self.head_gender, self.head_color1]:
            for param in head.parameters():
                param.requires_grad = False
                
    def unfreeze_stage1(self):
        """Unfreeze the backbone and tabular heads."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for head in [self.head_type, self.head_furlength, self.head_maturitysize, 
                     self.head_breed1, self.head_health, self.head_vaccinated,
                     self.head_dewormed, self.head_sterilized, self.head_gender, self.head_color1]:
            for param in head.parameters():
                param.requires_grad = True

    def forward(self, x, stage: int = 2):
        features = self.backbone(x)

        if self.extract_features:
            return features

        # Stage 1: Tabular Predictions
        out_type = self.head_type(features)
        out_fur = self.head_furlength(features)
        out_maturity = self.head_maturitysize(features)
        out_breed1 = self.head_breed1(features)
        out_health = self.head_health(features)
        out_vaccinated = self.head_vaccinated(features)
        out_dewormed = self.head_dewormed(features)
        out_sterilized = self.head_sterilized(features)
        out_gender = self.head_gender(features)
        out_color1 = self.head_color1(features)

        tabular_outputs = {
            'Type': out_type,
            'FurLength': out_fur,
            'MaturitySize': out_maturity,
            'Breed1': out_breed1,
            'Health': out_health,
            'Vaccinated': out_vaccinated,
            'Dewormed': out_dewormed,
            'Sterilized': out_sterilized,
            'Gender': out_gender,
            'Color1': out_color1,
        }

        if stage == 1:
            return tabular_outputs

        # Stage 2: Concatenate and predict AdoptionSpeed
        concat_features = torch.cat([
            features, out_type, out_fur, out_maturity, out_breed1,
            out_health, out_vaccinated, out_dewormed, out_sterilized,
            out_gender, out_color1
        ], dim=1)

        out_adoption = self.head_adoption_speed(concat_features)

        return {**tabular_outputs, 'AdoptionSpeed': out_adoption}
