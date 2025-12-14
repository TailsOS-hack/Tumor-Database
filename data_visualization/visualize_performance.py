import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TUMOR_DATA_DIR = os.path.join(DATA_DIR, 'brain_tumor', 'Testing')
ALZ_DATA_DIR = os.path.join(DATA_DIR, 'alzheimers') # Using whole dataset as no test split exists
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'brain_tumor_classifier.pt')
ALZ_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'alzheimers_classifier.pt')

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

# --- Loading Models ---

def load_models():
    print("Loading models...")
    
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
        
    return tumor_model, tumor_classes, alz_model

# --- Evaluation Logic ---

def predict(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.topk(probs, 1)
        return top_idx.item(), top_prob.item() * 100.0

# We need a custom way to handle different transforms for the two models on the same image.
class RawImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target, path

def run_evaluation():
    tumor_model, tumor_classes, alz_model = load_models()
    
    if not tumor_model or not alz_model:
        print("Both models are required for full evaluation.")
        return

    tumor_tfm = get_tumor_transform()
    alz_tfm = get_alzheimers_transform()

    # Data Containers
    combined_results = []
    tumor_only_results = []
    alz_only_results = []

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
            
            # Tumor Prediction
            t_tensor = tumor_tfm(img).unsqueeze(0).to(DEVICE)
            t_idx, t_conf = predict(tumor_model, t_tensor)
            t_pred = tumor_classes[t_idx]
            
            tumor_only_results.append({
                'True': true_label,
                'Pred': t_pred,
                'Conf': t_conf,
                'Correct': true_label == t_pred
            })

            # Alz Prediction (Cross-domain test)
            a_tensor = alz_tfm(img).unsqueeze(0).to(DEVICE)
            a_idx, a_conf = predict(alz_model, a_tensor)
            a_pred = ALZ_CLASSES[a_idx]

            # Combined Logic
            if a_conf > t_conf:
                winner_pred = a_pred
                winner_conf = a_conf
                model_used = 'Alzheimer'
            else:
                winner_pred = t_pred
                winner_conf = t_conf
                model_used = 'Tumor'
            
            combined_results.append({
                'True': true_label,
                'Pred': winner_pred,
                'Conf': winner_conf,
                'Correct': true_label == winner_pred, # Only correct if it picks the right tumor class
                'Model': model_used,
                'Dataset': 'Tumor'
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
            
            # Alz Prediction
            a_tensor = alz_tfm(img).unsqueeze(0).to(DEVICE)
            a_idx, a_conf = predict(alz_model, a_tensor)
            a_pred = ALZ_CLASSES[a_idx]
            
            alz_only_results.append({
                'True': true_label,
                'Pred': a_pred,
                'Conf': a_conf,
                'Correct': true_label == a_pred
            })

            # Tumor Prediction (Cross-domain test)
            t_tensor = tumor_tfm(img).unsqueeze(0).to(DEVICE)
            t_idx, t_conf = predict(tumor_model, t_tensor)
            t_pred = tumor_classes[t_idx]

            # Combined Logic
            if a_conf > t_conf:
                winner_pred = a_pred
                winner_conf = a_conf
                model_used = 'Alzheimer'
            else:
                winner_pred = t_pred
                winner_conf = t_conf
                model_used = 'Tumor'
            
            combined_results.append({
                'True': true_label,
                'Pred': winner_pred,
                'Conf': winner_conf,
                'Correct': true_label == winner_pred,
                'Model': model_used,
                'Dataset': 'Alzheimer'
            })

    return pd.DataFrame(tumor_only_results), pd.DataFrame(alz_only_results), pd.DataFrame(combined_results), tumor_classes

# --- Visualization ---

def plot_confusion_matrix(df, classes, title, filename):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(df['True'], df['Pred'], labels=classes)
    
    # Normalize to percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
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
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("Starting Performance Visualization (20% Subsample)...")
    df_tumor, df_alz, df_combined, tumor_classes = run_evaluation()
    
    output_dir = os.path.dirname(__file__)
    
    # Set style
    sns.set_style("darkgrid")
    
    # 1. Tumor Model Visualizations
    if not df_tumor.empty:
        print("Generating Tumor Model Plots...")
        plot_confusion_matrix(df_tumor, tumor_classes, "Confusion Matrix: Brain Tumor Model", os.path.join(output_dir, "tumor_confusion_matrix.png"))
        plot_accuracy_bar(df_tumor, "Accuracy by Class: Brain Tumor Model", os.path.join(output_dir, "tumor_accuracy_bar.png") )
    
    # 2. Alzheimer Model Visualizations
    if not df_alz.empty:
        print("Generating Alzheimer Model Plots...")
        plot_confusion_matrix(df_alz, ALZ_CLASSES, "Confusion Matrix: Alzheimer's Model", os.path.join(output_dir, "alz_confusion_matrix.png"))
        plot_accuracy_bar(df_alz, "Accuracy by Class: Alzheimer's Model", os.path.join(output_dir, "alz_accuracy_bar.png") )

    # 3. Combined Model Visualizations
    if not df_combined.empty:
        print("Generating Combined System Plots...")
        
        # Combined Confusion Matrix (All Classes)
        all_classes = sorted(list(set(tumor_classes + ALZ_CLASSES)))
        plot_confusion_matrix(df_combined, all_classes, "Confusion Matrix: Combined System (Competitive)", os.path.join(output_dir, "combined_confusion_matrix.png"))
        
        # Combined Accuracy
        plot_accuracy_bar(df_combined, "Accuracy by Class: Combined System", os.path.join(output_dir, "combined_accuracy_bar.png") )
        
        # Overall Accuracy Metric
        overall_acc = df_combined['Correct'].mean() * 100
        print(f"\nOverall Combined System Accuracy: {overall_acc:.2f}%")
        
        # Model Selection Accuracy
        df_combined['Correct_Model_Type'] = (
            ((df_combined['Dataset'] == 'Tumor') & (df_combined['Model'] == 'Tumor')) |
            ((df_combined['Dataset'] == 'Alzheimer') & (df_combined['Model'] == 'Alzheimer'))
        )
        model_selection_acc = df_combined['Correct_Model_Type'].mean() * 100
        print(f"Model Selection Accuracy (Tumor vs Alz): {model_selection_acc:.2f}%")

    print("\nVisualization Complete! Check the 'data_visualization' folder for images.")

if __name__ == "__main__":
    main()