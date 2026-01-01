import torch
import torch.nn as nn
from torchvision import models

class GatekeeperClassifier(nn.Module):
    def __init__(self, num_classes=3, freeze_base=True):
        super(GatekeeperClassifier, self).__init__()
        
        # Load pretrained ResNet50
        # 'weights' parameter is the modern way, but 'pretrained=True' is still often supported.
        # We'll use the modern weights enum if available, or fall back for broader compatibility.
        try:
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except:
            self.base_model = models.resnet50(pretrained=True)

        # Freeze the base layers if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Modify the final fully connected layer
        # ResNet50 fc layer input features is 2048
        num_ftrs = self.base_model.fc.in_features
        
        # We replace it with a Sequential block for multi-class classification
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        return self.base_model(x)
