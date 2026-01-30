import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import numpy as np

# Paths
GT_PATH = "data/evaluation/ground_truth/radiologist_test_key.csv"
RAD_PATH = "data/evaluation/radiologist_results/Dementia_score_sheet_REVISED (1).xlsx"
MODEL_PATH = "data/evaluation/model_results/model_predictions.csv"
OUTPUT_DIR = "data_visualization/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
print("Loading data...")
gt_df = pd.read_csv(GT_PATH)
rad_df = pd.read_excel(RAD_PATH)
model_df = pd.read_csv(MODEL_PATH)

# Clean Rad DataFrame
# Find the column that contains the diagnosis
rad_cols = [c for c in rad_df.columns if "Diagnosis" in c]
if not rad_cols:
    raise ValueError("Could not find diagnosis column in Radiologist file")
rad_col = rad_cols[0]
rad_df.rename(columns={rad_col: 'RadPrediction'}, inplace=True)

# Merge all
# Ensure FileNames match (strip whitespace)
gt_df['FileName'] = gt_df['FileName'].str.strip()
rad_df['FileName'] = rad_df['FileName'].str.strip()
model_df['FileName'] = model_df['FileName'].str.strip()

df = gt_df.merge(rad_df, on="FileName").merge(model_df, on="FileName")

print(f"Merged Data Shape: {df.shape}")

# Map Rad Labels
rad_map = {
    'NonDemented': 'NonDemented',
    'VeryMild': 'VeryMildDemented',
    'Mild': 'MildDemented',
    'Moderate': 'ModerateDemented'
}
df['RadPrediction_Mapped'] = df['RadPrediction'].map(rad_map)

# Check for unmapped values
if df['RadPrediction_Mapped'].isnull().any():
    print("Warning: Some Radiologist predictions could not be mapped:")
    print(df[df['RadPrediction_Mapped'].isnull()]['RadPrediction'].unique())

# Standard Classes
classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
# Short names for plotting
class_names_short = ['Normal', 'Very Mild', 'Mild', 'Moderate']

# --- Metrics ---
rad_acc = accuracy_score(df['TrueCategory'], df['RadPrediction_Mapped'])
model_acc = accuracy_score(df['TrueCategory'], df['ModelPrediction'])

print(f"Radiologist Accuracy: {rad_acc:.4f}")
print(f"AI Model Accuracy:    {model_acc:.4f}")

# --- Visualization 1: Accuracy Bar Chart ---
plt.figure(figsize=(10, 6))
# Set style
sns.set_style("whitegrid")

# Data for plot
acc_data = pd.DataFrame({
    'Participant': ['Radiologist', 'AI Model'],
    'Accuracy': [rad_acc, model_acc]
})

colors = ['#3498db', '#e74c3c'] # Blue, Red
ax = sns.barplot(x='Participant', y='Accuracy', data=acc_data, palette=colors)

plt.title('Diagnostic Accuracy on Test Set (n=100)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('')
plt.ylim(0, 1.1)

# Add value labels
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.02,
            f'{height:.1%}',
            ha="center", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=300)
print(f"Saved accuracy chart to {OUTPUT_DIR}/accuracy_comparison.png")
plt.close()

# --- Visualization 2: Side-by-Side Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Rad CM
cm_rad = confusion_matrix(df['TrueCategory'], df['RadPrediction_Mapped'], labels=classes)
sns.heatmap(cm_rad, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[0], cbar=False,
            annot_kws={"size": 14})
axes[0].set_title(f'Radiologist Confusion Matrix\nAccuracy: {rad_acc:.1%}', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Model CM
# Handle "Tumor_Predicted" if present by excluding from matrix but noting it
# Check if any predictions are not in classes
invalid_preds = df[~df['ModelPrediction'].isin(classes)]
if not invalid_preds.empty:
    print(f"Warning: Model predicted classes outside target set: {invalid_preds['ModelPrediction'].unique()}")
    # We will just compute CM on the standard classes.
    # Mismatches (Tumor) will count as "None" in the matrix row sums if we use labels=classes?
    # Actually, confusion_matrix sums over the predicted labels provided in 'labels'. 
    # If a true label is in 'labels' but pred is NOT, it is counted as a miss but not shown in a column?
    # Let's verifying sk-learn behavior: rows are True, cols are Pred.
    # If pred is not in labels, it's just not in any column.
    pass

cm_model = confusion_matrix(df['TrueCategory'], df['ModelPrediction'], labels=classes)
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[1], cbar=False,
            annot_kws={"size": 14})
axes[1].set_title(f'AI Model Confusion Matrix\nAccuracy: {model_acc:.1%}', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.suptitle("Confusion Matrix Comparison", fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_comparison.png", dpi=300)
print(f"Saved confusion matrices to {OUTPUT_DIR}/confusion_matrix_comparison.png")
plt.close()

# --- Visualization 3: Per-Class Accuracy (Sensitivity/Recall) ---
# Calculate recall per class
from sklearn.metrics import recall_score
recall_rad = recall_score(df['TrueCategory'], df['RadPrediction_Mapped'], labels=classes, average=None)
recall_model = recall_score(df['TrueCategory'], df['ModelPrediction'], labels=classes, average=None)

per_class_df = pd.DataFrame({
    'Class': class_names_short * 2,
    'Recall': np.concatenate([recall_rad, recall_model]),
    'Predictor': ['Radiologist'] * 4 + ['AI Model'] * 4
})

plt.figure(figsize=(12, 6))
sns.barplot(x='Class', y='Recall', hue='Predictor', data=per_class_df, palette=['#3498db', '#e74c3c'])
plt.title('Per-Class Sensitivity (Recall)', fontsize=16, fontweight='bold')
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.ylim(0, 1.1)
plt.legend(loc='lower right', title='Predictor')

# Annotate
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.0f%%')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_sensitivity.png", dpi=300)
print(f"Saved per-class sensitivity to {OUTPUT_DIR}/per_class_sensitivity.png")
plt.close()

print("Analysis Complete.")
