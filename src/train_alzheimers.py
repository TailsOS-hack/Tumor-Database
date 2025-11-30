import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time

# Configuration
DATA_DIR = 'data/alzheimers'
MODEL_SAVE_PATH = 'models/alzheimers_classifier.pt'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 4
# Check for CUDA (unlikely here) or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reduced image size for speed on CPU
IMG_SIZE = 160 

def get_data_loaders():
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load full dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Note: On Windows, num_workers=0 is safest. Increase to 2 or 4 if you want to try for more speed, 
    # but it can cause "BrokenPipe" errors.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, full_dataset.classes

def train_model(num_epochs_to_add=5):
    print(f"Using device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    
    train_loader, val_loader, class_names = get_data_loaders()
    print(f"Classes: {class_names}")

    # Load model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        model = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    else:
        print("Initializing new MobileNetV3-Large model...")
        # MobileNetV3 is much faster than ResNet18
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        # Modify the classifier head
        # MobileNetV3 classifier is a Sequential block. The last layer is '3'.
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
        model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training for {num_epochs_to_add} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs_to_add):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs_to_add}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # Progress tracking
            total_batches = len(dataloader)
            print_interval = max(1, total_batches // 5) # Print progress 5 times per epoch

            # Iterate over data
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and (i + 1) % print_interval == 0:
                     print(f"  Batch {i+1}/{total_batches} processed...")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        epoch_duration = time.time() - epoch_start
        print(f"Epoch completed in {epoch_duration:.0f}s")

    total_time = time.time() - start_time
    print(f"Training block complete in {total_time//60:.0f}m {total_time%60:.0f}s. Validation Accuracy: {epoch_acc:.4f}")
    
    # Save model
    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model, MODEL_SAVE_PATH)

if __name__ == '__main__':
    train_model(5)

