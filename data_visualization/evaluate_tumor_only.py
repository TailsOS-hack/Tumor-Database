import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_DIR = os.path.join(DATA_DIR, 'brain_tumor', 'Testing')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'brain_tumor_classifier.pt')
OUTPUT_DIR = os.path.dirname(__file__)

# Classes the model was trained on (excluding 'notumor')
TARGET_CLASSES = ['glioma', 'meningioma', 'pituitary']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def get_transforms():
    # Matches src/train_complete_suite.py
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Reconstruct the model architecture
        model = models.efficientnet_b3(weights=None)
        # The training script replaces the classifier[1] with a Linear layer of size 3
        # original in_features for B3 is 1536
        in_features = model.classifier[1].in_features 
        model.classifier[1] = nn.Linear(in_features, len(TARGET_CLASSES))
        
        # Load the state dictionary
        # Note: The training script saves the *whole model* object (torch.save(model, path)), 
        # not just state_dict. Let's try loading the whole object first.
        try:
            model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except:
            # Fallback if it was saved as state_dict
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict)
            
        model.eval().to(DEVICE)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate():
    model = load_model()
    if model is None:
        return

    transform = get_transforms()
    
    # Custom loading to filter out 'notumor'
    print(f"Loading test data from {TEST_DIR}...")
    
    images = []
    labels = []
    
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
    
    for cls_name in TARGET_CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory {cls_dir} not found.")
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Found {len(files)} images for class '{cls_name}'")
        
        for f in files:
            images.append(os.path.join(cls_dir, f))
            labels.append(class_to_idx[cls_name])
            
    if not images:
        print("No images found for evaluation.")
        return

    print(f"Total images to evaluate: {len(images)}")
    
    # Inference
    y_true = []
    y_pred = []
    probs = [] # For potential ROC if needed
    
    with torch.no_grad():
        for img_path, label in tqdm(zip(images, labels), total=len(images)):
            try:
                # Load image using PIL (as per training script)
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                y_true.append(label)
                y_pred.append(predicted.item())
                probs.append(probabilities.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    report = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(df_report)
    
    # --- Visualization ---
    sns.set_style("whitegrid")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # Calculate row-normalized percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title('Confusion Matrix: Tumor Size Classification', fontweight='bold')
    plt.ylabel('True Size', fontweight='bold')
    plt.xlabel('Predicted Size', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'tumor_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # 2. Accuracy Bar Chart
    # Calculate accuracy per class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class_acc = cm_norm.diagonal() * 100
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=TARGET_CLASSES, y=per_class_acc, palette='viridis')
    plt.title('Accuracy of Tumor Size Prediction')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Tumor Size')
    plt.ylim(0, 100)
    for i, v in enumerate(per_class_acc):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, 'tumor_accuracy_bar.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved Accuracy Bar Chart to {acc_path}")

    # 3. Detailed Metrics Grouped Bar (Precision, Recall, F1)
    metrics_data = []
    for cls in TARGET_CLASSES:
        metrics_data.append({'Class': cls, 'Metric': 'Precision', 'Value': report[cls]['precision'] * 100})
        metrics_data.append({'Class': cls, 'Metric': 'Recall', 'Value': report[cls]['recall'] * 100})
        metrics_data.append({'Class': cls, 'Metric': 'F1-Score', 'Value': report[cls]['f1-score'] * 100})
    
    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x='Class', y='Value', hue='Metric', palette='muted')
    plt.title('Performance Metrics by Tumor Size')
    plt.ylabel('Score (%)')
    plt.xlabel('Tumor Size')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    metrics_path = os.path.join(OUTPUT_DIR, 'tumor_detailed_metrics.png')
    plt.savefig(metrics_path)
    plt.close()
    print(f"Saved Detailed Metrics Chart to {metrics_path}")

    # 4. ROC Curves (One-vs-Rest)
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(len(TARGET_CLASSES)))
    n_classes = y_true_bin.shape[1]
    y_score = np.array(probs)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve for {TARGET_CLASSES[i]} Size (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Curves - Tumor Size Prediction', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=16)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, 'tumor_roc_curves.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC Curves to {roc_path}")

if __name__ == "__main__":
    evaluate()
