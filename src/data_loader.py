import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class BinaryGatekeeperDataset(Dataset):
    """
    A custom dataset for binary classification:
    Class 0: Tumor Group (data/brain_tumor)
    Class 1: Dementia Group (data/alzheimers)
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32) # Float for BCEWithLogitsLoss
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or handle gracefully (skipping is harder in __getitem__)
            # For now, we'll try to return the next valid item or a black image
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)

def get_transforms():
    """
    Returns training and validation transforms.
    """
    # Augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) # ImageNet standards
    ])

    # No augmentation for validation, just resizing and normalization
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms

def load_data(root_dir='data', test_split=0.2, batch_size=32, num_workers=0):
    """
    Loads data from directories, creates datasets and dataloaders.
    
    Args:
        root_dir: Project root containing the 'data' folder.
        test_split: Fraction of data to use for validation.
    """
    tumor_dir = os.path.join(root_dir, 'brain_tumor')
    alzheimers_dir = os.path.join(root_dir, 'alzheimers')

    # Gather all file paths
    # Recursive search for images
    tumor_images = glob.glob(os.path.join(tumor_dir, '**', '*.jpg'), recursive=True) + \
                   glob.glob(os.path.join(tumor_dir, '**', '*.png'), recursive=True) + \
                   glob.glob(os.path.join(tumor_dir, '**', '*.jpeg'), recursive=True)
    
    alz_images = glob.glob(os.path.join(alzheimers_dir, '**', '*.jpg'), recursive=True) + \
                 glob.glob(os.path.join(alzheimers_dir, '**', '*.png'), recursive=True) + \
                 glob.glob(os.path.join(alzheimers_dir, '**', '*.jpeg'), recursive=True)

    print(f"Found {len(tumor_images)} Tumor images.")
    print(f"Found {len(alz_images)} Alzheimer's images.")

    # Balance the dataset (Undersampling majority class)
    import random
    random.seed(42)
    
    if len(tumor_images) < len(alz_images):
        print(f"Undersampling Alzheimer's class to match Tumor class ({len(tumor_images)} samples).")
        alz_images = random.sample(alz_images, len(tumor_images))
    else:
        print(f"Undersampling Tumor class to match Alzheimer's class ({len(alz_images)} samples).")
        tumor_images = random.sample(tumor_images, len(alz_images))
    
    print(f"Balanced Dataset: {len(tumor_images)} Tumor, {len(alz_images)} Alzheimer's")

    # Create lists of paths and labels
    all_paths = tumor_images + alz_images
    # Label 0 for Tumor, 1 for Alzheimers
    all_labels = [0] * len(tumor_images) + [1] * len(alz_images)

    # Split into train and validation sets (using indices)
    total_len = len(all_paths)
    val_len = int(total_len * test_split)
    train_len = total_len - val_len
    
    # We need to shuffle before splitting because we concatenated the lists
    # Using random_split is easier if we wrap the data first, but we need different transforms
    # So we will split the indices first.
    
    full_dataset_indices = list(range(total_len))
    # Shuffle handled by random_split or manually
    train_indices, val_indices = random_split(full_dataset_indices, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    
    train_paths = [all_paths[i] for i in train_indices.indices]
    train_labels = [all_labels[i] for i in train_indices.indices]
    
    val_paths = [all_paths[i] for i in val_indices.indices]
    val_labels = [all_labels[i] for i in val_indices.indices]

    # Get transforms
    train_tf, val_tf = get_transforms()

    # Create Datasets
    train_dataset = BinaryGatekeeperDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = BinaryGatekeeperDataset(val_paths, val_labels, transform=val_tf)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
