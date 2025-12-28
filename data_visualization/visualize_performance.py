import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import warnings
import random
import sys

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gatekeeper_model import GatekeeperClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TUMOR_DATA_DIR = os.path.join(DATA_DIR, 'brain_tumor', 'Testing')
ALZ_DATA_DIR = os.path.join(DATA_DIR, 'alzheimers') # Using whole dataset as no test split exists
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'brain_tumor_classifier.pt')
ALZ_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'alzheimers_classifier.pt')
GATEKEEPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gatekeeper_classifier.pt')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model Definitions (Copied from src/radiology_report_gui.py) ---

ALZ_CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def build_tumor_model(arch: str, num_classes: int):
    if arch == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def get_tumor_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(320, antialias=True),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_alzheimers_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((160, 160), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_gatekeeper_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# --- Loading Models ---

def load_models():
    print("Loading models...")
    
    # Load Gatekeeper
    gatekeeper_model = None
    if os.path.exists(GATEKEEPER_MODEL_PATH):
        try:
            gatekeeper_model = GatekeeperClassifier(freeze_base=False)
            gatekeeper_model.load_state_dict(torch.load(GATEKEEPER_MODEL_PATH, map_location=DEVICE, weights_only=True))
            gatekeeper_model.eval().to(DEVICE)
            print("Gatekeeper Model Loaded.")
        except Exception as e:
            print(f"Error loading gatekeeper model: {e}")
    else:
        print(f"Gatekeeper model not found at {GATEKEEPER_MODEL_PATH}")

    # Load Tumor Model
    tumor_model = None
    tumor_classes = []
    if os.path.exists(TUMOR_MODEL_PATH):
        try:
            checkpoint = torch.load(TUMOR_MODEL_PATH, map_location=DEVICE)
            arch = checkpoint.get("arch", "efficientnet_b3")
            tumor_classes = checkpoint.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
            tumor_model = build_tumor_model(arch, len(tumor_classes))
            tumor_model.load_state_dict(checkpoint["model_state"])
            tumor_model.eval().to(DEVICE)
            print(f"Tumor Model Loaded. Classes: {tumor_classes}")
        except Exception as e:
            print(f"Error loading tumor model: {e}")
    else:
        print(f"Tumor model not found at {TUMOR_MODEL_PATH}")

    # Load Alzheimer's Model
    alz_model = None
    if os.path.exists(ALZ_MODEL_PATH):
        try:
            alz_model = torch.load(ALZ_MODEL_PATH, map_location=DEVICE, weights_only=False)
            alz_model.eval().to(DEVICE)
            print(f"Alzheimer's Model Loaded. Classes: {ALZ_CLASSES}")
        except Exception as e:
            print(f"Error loading alzheimer model: {e}")
    else:
        print(f"Alzheimer's model not found at {ALZ_MODEL_PATH}")
        
    return gatekeeper_model, tumor_model, tumor_classes, alz_model

# --- Evaluation Logic ---

def predict_specialized(model, image_tensor, classes):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.topk(probs, 1)
        return classes[top_idx.item()], top_prob.item() * 100.0

def predict_gatekeeper(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        # > 0.5 is Dementia (Class 1), <= 0.5 is Tumor (Class 0)
        if prob > 0.5:
            return "dementia", prob
        else:
            return "tumor", 1 - prob

# We need a custom way to handle different transforms for the two models on the same image.
class RawImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target, path

def run_evaluation():
    gatekeeper_model, tumor_model, tumor_classes, alz_model = load_models()
    
    if not (gatekeeper_model and tumor_model and alz_model):
        print("All three models are required for full evaluation.")
        return

    tumor_tfm = get_tumor_transform()
    alz_tfm = get_alzheimers_transform()
    gate_tfm = get_gatekeeper_transform()

    # Data Containers
    combined_results = []

    # 1. Evaluate on Tumor Dataset
    print(f"Evaluating on Tumor Dataset ({TUMOR_DATA_DIR})...")
    if os.path.exists(TUMOR_DATA_DIR):
        tumor_ds = RawImageFolder(TUMOR_DATA_DIR)
        
        # Subsample 20%
        random.seed(42)
        random.shuffle(tumor_ds.samples)
        original_count = len(tumor_ds.samples)
        tumor_ds.samples = tumor_ds.samples[:int(original_count * 0.2)]
        print(f"  > Using 20% random subsample: {len(tumor_ds)} images (from {original_count})")
        
        for img, label_idx, path in tqdm(tumor_ds):
            true_label = tumor_ds.classes[label_idx]
            
            # 1. Gatekeeper Routing
            g_tensor = gate_tfm(img).unsqueeze(0).to(DEVICE)
            domain, g_conf = predict_gatekeeper(gatekeeper_model, g_tensor)
            
            final_pred = "Unknown"
            model_used = "Unknown"

            if domain == "tumor":
                # Routed Correctly to Tumor Model
                t_tensor = tumor_tfm(img).unsqueeze(0).to(DEVICE)
                final_pred, _ = predict_specialized(tumor_model, t_tensor, tumor_classes)
                model_used = "Tumor"
            else:
                # Routed Incorrectly to Alzheimer's Model
                a_tensor = alz_tfm(img).unsqueeze(0).to(DEVICE)
                final_pred, _ = predict_specialized(alz_model, a_tensor, ALZ_CLASSES)
                model_used = "Alzheimer"
            
            combined_results.append({
                'True': true_label,
                'Pred': final_pred,
                'Correct': true_label == final_pred,
                'Model': model_used,
                'Dataset': 'Tumor',
                'Routed_To': domain
            })

    # 2. Evaluate on Alzheimer Dataset
    print(f"Evaluating on Alzheimer Dataset ({ALZ_DATA_DIR})...")
    if os.path.exists(ALZ_DATA_DIR):
        alz_ds = RawImageFolder(ALZ_DATA_DIR)
        
        # Subsample 20%
        random.seed(42)
        random.shuffle(alz_ds.samples)
        original_count = len(alz_ds.samples)
        alz_ds.samples = alz_ds.samples[:int(original_count * 0.2)]
        print(f"  > Using 20% random subsample: {len(alz_ds)} images (from {original_count})")
        
        for img, label_idx, path in tqdm(alz_ds):
            true_label = alz_ds.classes[label_idx]
            
            # 1. Gatekeeper Routing
            g_tensor = gate_tfm(img).unsqueeze(0).to(DEVICE)
            domain, g_conf = predict_gatekeeper(gatekeeper_model, g_tensor)
            
            final_pred = "Unknown"
            model_used = "Unknown"

            if domain == "dementia":
                # Routed Correctly to Alz Model
                a_tensor = alz_tfm(img).unsqueeze(0).to(DEVICE)
                final_pred, _ = predict_specialized(alz_model, a_tensor, ALZ_CLASSES)
                model_used = "Alzheimer"
            else:
                # Routed Incorrectly to Tumor Model
                t_tensor = tumor_tfm(img).unsqueeze(0).to(DEVICE)
                final_pred, _ = predict_specialized(tumor_model, t_tensor, tumor_classes)
                model_used = "Tumor"
            
            combined_results.append({
                'True': true_label,
                'Pred': final_pred,
                'Correct': true_label == final_pred,
                'Model': model_used,
                'Dataset': 'Alzheimer',
                'Routed_To': domain
            })

    return pd.DataFrame(combined_results), tumor_classes

# --- Visualization ---

def plot_confusion_matrix(df, classes, title, filename):
    plt.figure(figsize=(12, 10))
    # Filter classes to only those present in the dataframe + expected classes
    unique_true = df['True'].unique()
    unique_pred = df['Pred'].unique()
    present_classes = sorted(list(set(unique_true) | set(unique_pred) | set(classes)))
    
    cm = confusion_matrix(df['True'], df['Pred'], labels=present_classes)
    
    # Normalize to percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=present_classes, yticklabels=present_classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_accuracy_bar(df, title, filename):
    plt.figure(figsize=(10, 6))
    acc_by_class = df.groupby('True')['Correct'].mean() * 100
    sns.barplot(x=acc_by_class.index, y=acc_by_class.values, palette='viridis')
    plt.title(title)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("Starting Hierarchical Performance Visualization (20% Subsample)...")
    df_combined, tumor_classes = run_evaluation()
    
    output_dir = os.path.dirname(__file__)
    
    # Set style
    sns.set_style("darkgrid")
    
    if not df_combined.empty:
        print("Generating Combined System Plots...")
        
        # Combined Confusion Matrix (All Classes)
        all_classes = sorted(list(set(tumor_classes + ALZ_CLASSES)))
        plot_confusion_matrix(df_combined, all_classes, "Confusion Matrix: Hierarchical System (Gatekeeper)", os.path.join(output_dir, "hierarchical_confusion_matrix.png"))
        
        # Combined Accuracy
        plot_accuracy_bar(df_combined, "Accuracy by Class: Hierarchical System", os.path.join(output_dir, "hierarchical_accuracy_bar.png") )
        
        # Overall Accuracy Metric
        overall_acc = df_combined['Correct'].mean() * 100
        print(f"\nOverall Hierarchical System Accuracy: {overall_acc:.2f}%")
        
        # Gatekeeper Routing Accuracy
        # Correct routing means Tumor dataset -> 'tumor' domain AND Alzheimer dataset -> 'dementia' domain
        df_combined['Correct_Routing'] = (
            ((df_combined['Dataset'] == 'Tumor') & (df_combined['Routed_To'] == 'tumor')) |
            ((df_combined['Dataset'] == 'Alzheimer') & (df_combined['Routed_To'] == 'dementia'))
        )
        routing_acc = df_combined['Correct_Routing'].mean() * 100
        print(f"Gatekeeper Routing Accuracy: {routing_acc:.2f}%")

    print("\nVisualization Complete! Check the 'data_visualization' folder for images.")

if __name__ == "__main__":
    main()