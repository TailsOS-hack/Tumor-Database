import os
import glob
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# ==========================================
# Logging Setup
# ==========================================
# Configure logging to display time, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_ROOT = 'data'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# Dataset Class
# ==========================================
class MedicalImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return a blank image to prevent crashing
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(label, dtype=torch.long)

# ==========================================
# Helper Functions
# ==========================================
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def get_transforms():
    logger.info("Initializing data transforms...")
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def train_and_save(model, train_loader, val_loader, save_path, num_classes, model_name):
    logger.info(f"STARTING TRAINING: {model_name}")
    logger.info(f"Target Device: {DEVICE}")
    logger.info(f"Saving to: {save_path}")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Optional: Log batch progress for very large datasets
            # if (batch_idx + 1) % 10 == 0:
            #    logger.info(f"Batch {batch_idx+1}/{num_batches} processed")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Timing and ETA
        epoch_duration = time.time() - epoch_start_time
        elapsed_total = time.time() - total_start_time
        avg_epoch_time = elapsed_total / (epoch + 1)
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        logger.info(f"Results: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Acc={val_acc:.4f}")
        logger.info(f"Timing: Epoch={format_time(epoch_duration)} | Elapsed={format_time(elapsed_total)} | ETA={format_time(eta_seconds)}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, save_path)
            logger.info(f"New best model saved! (Acc: {best_acc:.4f})")

    logger.info(f"COMPLETED TRAINING: {model_name}. Total Time: {format_time(time.time() - total_start_time)}")

# ==========================================
# Data Gathering Logic
# ==========================================
def gather_files():
    logger.info("Gathering file paths from 'data' directory...")
    # Helper to get files
    def get_files(pattern):
        files = glob.glob(os.path.join(DATA_ROOT, pattern), recursive=True)
        return files

    # Brain Tumor Paths
    bt_train = os.path.join('brain_tumor', 'Training')
    bt_test = os.path.join('brain_tumor', 'Testing')
    
    # Tumor Subtypes
    glioma = get_files(os.path.join(bt_train, 'glioma', '*')) + get_files(os.path.join(bt_test, 'glioma', '*'))
    meningioma = get_files(os.path.join(bt_train, 'meningioma', '*')) + get_files(os.path.join(bt_test, 'meningioma', '*'))
    pituitary = get_files(os.path.join(bt_train, 'pituitary', '*')) + get_files(os.path.join(bt_test, 'pituitary', '*'))
    
    # Tumor Normal
    tumor_normal = get_files(os.path.join(bt_train, 'notumor', '*')) + get_files(os.path.join(bt_test, 'notumor', '*'))

    # Alzheimer's Paths
    alz = 'alzheimers'
    mild = get_files(os.path.join(alz, 'MildDemented', '*'))
    moderate = get_files(os.path.join(alz, 'ModerateDemented', '*'))
    very_mild = get_files(os.path.join(alz, 'VeryMildDemented', '*'))
    alz_normal = get_files(os.path.join(alz, 'NonDemented', '*'))
    
    logger.info(f"Found {len(glioma)} Glioma, {len(meningioma)} Meningioma, {len(pituitary)} Pituitary, {len(tumor_normal)} Tumor-Normal")
    logger.info(f"Found {len(mild)} Mild, {len(moderate)} Moderate, {len(very_mild)} VeryMild, {len(alz_normal)} Alz-Normal")

    return {
        'glioma': glioma, 'meningioma': meningioma, 'pituitary': pituitary,
        'tumor_normal': tumor_normal,
        'mild': mild, 'moderate': moderate, 'very_mild': very_mild,
        'alz_normal': alz_normal
    }

# ==========================================
# Main Execution
# ==========================================
def main():
    start_time = time.time()
    logger.info("=== STARTING UNIFIED TRAINING SUITE ===")
    
    files = gather_files()
    train_tf, val_tf = get_transforms()

    # --------------------------------------
    # 1. Train Gatekeeper (Normal vs Tumor vs Dementia)
    # --------------------------------------
    logger.info("\n=== STAGE 1/3: GATEKEEPER MODEL ===")
    
    gk_paths = []
    gk_labels = []

    # Normal (0)
    normals = files['tumor_normal'] + files['alz_normal']
    gk_paths.extend(normals)
    gk_labels.extend([0] * len(normals))

    # Tumor (1)
    tumors = files['glioma'] + files['meningioma'] + files['pituitary']
    gk_paths.extend(tumors)
    gk_labels.extend([1] * len(tumors))

    # Dementia (2)
    dementias = files['mild'] + files['moderate'] + files['very_mild']
    gk_paths.extend(dementias)
    gk_labels.extend([2] * len(dementias))
    
    logger.info(f"Data Prep: Normal={len(normals)}, Tumor={len(tumors)}, Dementia={len(dementias)}")
    
    # Split
    indices = list(range(len(gk_paths)))
    train_idx, val_idx = random_split(indices, [int(0.8*len(indices)), len(indices)-int(0.8*len(indices))])
    
    train_ds = MedicalImageDataset([gk_paths[i] for i in train_idx.indices], 
                                   [gk_labels[i] for i in train_idx.indices], transform=train_tf)
    val_ds = MedicalImageDataset([gk_paths[i] for i in val_idx.indices], 
                                 [gk_labels[i] for i in val_idx.indices], transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Gatekeeper Model (ResNet50)
    logger.info("Initializing Gatekeeper (ResNet50)...")
    from gatekeeper_model import GatekeeperClassifier
    gk_model = GatekeeperClassifier(num_classes=3)
    
    train_and_save(gk_model, train_loader, val_loader, 
                   os.path.join(MODELS_DIR, 'gatekeeper_classifier.pt'), num_classes=3, model_name="Gatekeeper")


    # --------------------------------------
    # 2. Train Tumor Model (Glioma vs Meningioma vs Pituitary)
    # --------------------------------------
    logger.info("\n=== STAGE 2/3: TUMOR SPECIALIST MODEL ===")
    
    tm_paths = []
    tm_labels = []
    
    tm_paths.extend(files['glioma'])
    tm_labels.extend([0] * len(files['glioma']))
    
    tm_paths.extend(files['meningioma'])
    tm_labels.extend([1] * len(files['meningioma']))
    
    tm_paths.extend(files['pituitary'])
    tm_labels.extend([2] * len(files['pituitary']))
    
    logger.info(f"Data Prep: Glioma={len(files['glioma'])}, Meningioma={len(files['meningioma'])}, Pituitary={len(files['pituitary'])}")

    indices = list(range(len(tm_paths)))
    train_idx, val_idx = random_split(indices, [int(0.8*len(indices)), len(indices)-int(0.8*len(indices))])
    
    train_ds = MedicalImageDataset([tm_paths[i] for i in train_idx.indices], 
                                   [tm_labels[i] for i in train_idx.indices], transform=train_tf)
    val_ds = MedicalImageDataset([tm_paths[i] for i in val_idx.indices], 
                                 [tm_labels[i] for i in val_idx.indices], transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Tumor Model (EfficientNet-B3)
    logger.info("Initializing Tumor Model (EfficientNet-B3)...")
    tm_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    tm_model.classifier[1] = nn.Linear(tm_model.classifier[1].in_features, 3)
    
    train_and_save(tm_model, train_loader, val_loader, 
                   os.path.join(MODELS_DIR, 'brain_tumor_classifier.pt'), num_classes=3, model_name="Tumor Specialist")


    # --------------------------------------
    # 3. Train Dementia Model (Mild vs Moderate vs VeryMild)
    # --------------------------------------
    logger.info("\n=== STAGE 3/3: DEMENTIA SPECIALIST MODEL ===")
    
    dm_paths = []
    dm_labels = []
    
    dm_paths.extend(files['mild'])
    dm_labels.extend([0] * len(files['mild']))
    
    dm_paths.extend(files['moderate'])
    dm_labels.extend([1] * len(files['moderate']))
    
    dm_paths.extend(files['very_mild'])
    dm_labels.extend([2] * len(files['very_mild']))
    
    logger.info(f"Data Prep: Mild={len(files['mild'])}, Moderate={len(files['moderate'])}, VeryMild={len(files['very_mild'])}")

    indices = list(range(len(dm_paths)))
    train_idx, val_idx = random_split(indices, [int(0.8*len(indices)), len(indices)-int(0.8*len(indices))])
    
    train_ds = MedicalImageDataset([dm_paths[i] for i in train_idx.indices], 
                                   [dm_labels[i] for i in train_idx.indices], transform=train_tf)
    val_ds = MedicalImageDataset([dm_paths[i] for i in val_idx.indices], 
                                 [dm_labels[i] for i in val_idx.indices], transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Dementia Model (MobileNetV3 Large)
    logger.info("Initializing Dementia Model (MobileNetV3)...")
    dm_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    dm_model.classifier[3] = nn.Linear(dm_model.classifier[3].in_features, 3)
    
    train_and_save(dm_model, train_loader, val_loader, 
                   os.path.join(MODELS_DIR, 'alzheimers_classifier.pt'), num_classes=3, model_name="Dementia Specialist")

    logger.info(f"\n=== ALL TRAINING COMPLETE. Total Suite Time: {format_time(time.time() - start_time)} ===")

if __name__ == '__main__':
    main()