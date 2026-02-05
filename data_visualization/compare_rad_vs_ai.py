import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import os
import numpy as np
import glob

# Paths
GT_PATH = "data/evaluation/ground_truth/radiologist_test_key.csv"
RAD_DIR = "data/evaluation/radiologist_results/"
MODEL_PATH = "data/evaluation/model_results/model_predictions.csv"
OUTPUT_DIR = "data_visualization/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard Classes
classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
class_names_short = ['Normal', 'Very Mild', 'Mild', 'Moderate']

# Map Rad Labels
rad_map = {
    'NonDemented': 'NonDemented',
    'VeryMild': 'VeryMildDemented',
    'Mild': 'MildDemented',
    'Moderate': 'ModerateDemented',
    'Non Demented': 'NonDemented',
    'Very Mild': 'VeryMildDemented',
    'Moderate Demented': 'ModerateDemented'
}

def load_clean_sheet(filepath, sheet_name):
    """Loads a specific sheet and returns a cleaned DataFrame."""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading {filepath} sheet {sheet_name}: {e}")
        return None

    # Find diagnosis column
    rad_cols = [c for c in df.columns if "Diagnosis" in c]
    if not rad_cols:
        return None
    
    rad_col = rad_cols[0]
    
    # Check if empty
    if df[rad_col].count() < 50: # Threshold to skip empty/template sheets
        return None

    df = df[['FileName', rad_col]].copy()
    df.rename(columns={rad_col: 'RadPrediction'}, inplace=True)
    df['FileName'] = df['FileName'].astype(str).str.strip()
    
    # Map predictions
    df['RadPrediction'] = df['RadPrediction'].astype(str).str.strip()
    df['RadPrediction_Mapped'] = df['RadPrediction'].map(rad_map)
    
    # Fallback for exact matches
    mask = df['RadPrediction_Mapped'].isna() & df['RadPrediction'].isin(classes)
    df.loc[mask, 'RadPrediction_Mapped'] = df.loc[mask, 'RadPrediction']

    return df

# 1. Load Ground Truth & Model
print("Loading GT and Model...")
gt_df = pd.read_csv(GT_PATH)
gt_df['FileName'] = gt_df['FileName'].str.strip()

model_df = pd.read_csv(MODEL_PATH)
model_df['FileName'] = model_df['FileName'].str.strip()

# Merge GT and Model
main_df = gt_df.merge(model_df, on="FileName")
model_acc = accuracy_score(main_df['TrueCategory'], main_df['ModelPrediction'])
print(f"AI Model Accuracy: {model_acc:.4f}")

# 2. Load All Radiologists
rad_files = glob.glob(os.path.join(RAD_DIR, "*.xlsx"))
rad_results = []

processed_ids = set()

print(f"Scanning {len(rad_files)} radiologist files...")

for fpath in rad_files:
    fname = os.path.basename(fpath)
    try:
        xls = pd.ExcelFile(fpath)
        for sheet in xls.sheet_names:
            r_df = load_clean_sheet(fpath, sheet)
            if r_df is not None:
                rad_id = sheet # Use sheet name as ID (Kalapos, Thamburaj, Kanekar)
                
                # Deduplicate if same ID found in multiple files (use the first one or valid one)
                if rad_id in processed_ids:
                    print(f"Skipping duplicate Rad ID: {rad_id} in {fname}")
                    continue
                
                # Merge with main_df to ensure alignment
                temp_df = main_df.merge(r_df, on="FileName", how="left")
                temp_df['RadPrediction_Mapped'] = temp_df['RadPrediction_Mapped'].fillna("Unknown")
                
                acc = accuracy_score(temp_df['TrueCategory'], temp_df['RadPrediction_Mapped'])
                print(f"Loaded {rad_id} from {fname}: Accuracy = {acc:.4f}")
                
                rad_results.append({
                    'id': rad_id,
                    'accuracy': acc,
                    'df': r_df
                })
                processed_ids.add(rad_id)
                
    except Exception as e:
        print(f"Error processing file {fname}: {e}")

# 3. Aggregation Logic
if not rad_results:
    print("No valid radiologist results found. Exiting.")
    exit()

# Sort by accuracy descending
rad_results.sort(key=lambda x: x['accuracy'], reverse=True)

# Top 2
top_2_rads = rad_results[:2]
if len(top_2_rads) >= 2:
    top_2_avg_acc = np.mean([r['accuracy'] for r in top_2_rads])
    print(f"Top 2 ({[r['id'] for r in top_2_rads]}) Avg Accuracy: {top_2_avg_acc:.4f}")
else:
    top_2_avg_acc = rad_results[0]['accuracy']

# All 3
all_accs = [r['accuracy'] for r in rad_results]
mean_all = np.mean(all_accs)
median_all = np.median(all_accs)

# Skew Check
skew_diff = abs(mean_all - median_all)
if skew_diff > 0.05: # > 5% diff
    final_all_metric = median_all
    all_metric_name = "All Rads (Median)"
    print(f"Data skewed (diff {skew_diff:.4f}). Using Median: {median_all:.4f}")
else:
    final_all_metric = mean_all
    all_metric_name = "All Rads (Mean)"
    print(f"Data not skewed (diff {skew_diff:.4f}). Using Mean: {mean_all:.4f}")

# 4. Visualization
# A. Accuracy Comparison Bar Chart
viz_data = []
# Individual Rads
for r in rad_results:
    viz_data.append({'Participant': r['id'], 'Accuracy': r['accuracy'], 'Type': 'Individual'})

# Aggregates
if len(rad_results) > 1:
    viz_data.append({'Participant': 'Top 2 Rads (Avg)', 'Accuracy': top_2_avg_acc, 'Type': 'Aggregate'})
    viz_data.append({'Participant': all_metric_name, 'Accuracy': final_all_metric, 'Type': 'Aggregate'})

# AI
viz_data.append({'Participant': 'AI Model', 'Accuracy': model_acc, 'Type': 'AI'})

viz_df = pd.DataFrame(viz_data)

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
# Palette: Rads=Blues, Agg=Greens, AI=Red
palette = {}
for i, row in viz_df.iterrows():
    if row['Type'] == 'Individual': palette[row['Participant']] = '#3498db'
    elif row['Type'] == 'Aggregate': palette[row['Participant']] = '#2ecc71'
    elif row['Type'] == 'AI': palette[row['Participant']] = '#e74c3c'

ax = sns.barplot(x='Participant', y='Accuracy', data=viz_df, palette=palette)
plt.title('Diagnostic Accuracy Comparison: Radiologists vs AI', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=15)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=300)
plt.close()

# B. Confusion Matrices
# We assume up to 4 plots (3 Rads + 1 AI)
num_rads = len(rad_results)
total_plots = num_rads + 1
cols = 2
rows = (total_plots + 1) // 2

fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows))
axes = axes.flatten()

# Plot Rads
for i, r in enumerate(rad_results):
    temp_df = main_df.merge(r['df'], on="FileName", how="left")
    temp_df['RadPrediction_Mapped'] = temp_df['RadPrediction_Mapped'].fillna("Unknown")
    
    # We only plot confusion for valid classes. Unknowns will be ignored by sklearn if not in labels.
    # To see them, we'd need to add 'Unknown' to labels, but standard CM is usually strictly on classes.
    # Accuracy score reflects the penalty.
    cm = confusion_matrix(temp_df['TrueCategory'], temp_df['RadPrediction_Mapped'], labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[i], cbar=False,
                annot_kws={"size": 14})
    axes[i].set_title(f"{r['id']}\nAccuracy: {r['accuracy']:.1%}", fontsize=14, fontweight='bold')
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')

# Plot AI (Next slot)
ai_idx = num_rads
cm_model = confusion_matrix(main_df['TrueCategory'], main_df['ModelPrediction'], labels=classes)
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[ai_idx], cbar=False,
            annot_kws={"size": 14})
axes[ai_idx].set_title(f"AI Model\nAccuracy: {model_acc:.1%}", fontsize=14, fontweight='bold')
axes[ai_idx].set_ylabel('True Label')
axes[ai_idx].set_xlabel('Predicted Label')

# Hide unused subplots
for j in range(ai_idx + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Confusion Matrix Comparison", fontsize=18)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_comparison.png", dpi=300)
plt.close()

# C. Per-Class Sensitivity (Recall)
# Compare: AI, Best Rad, Avg Rad
recall_ai = recall_score(main_df['TrueCategory'], main_df['ModelPrediction'], labels=classes, average=None)

# Best Rad
best_rad = rad_results[0]
temp_best = main_df.merge(best_rad['df'], on="FileName", how="left")
temp_best['RadPrediction_Mapped'] = temp_best['RadPrediction_Mapped'].fillna("Unknown")
recall_best = recall_score(temp_best['TrueCategory'], temp_best['RadPrediction_Mapped'], labels=classes, average=None)

# Avg Rad
all_recalls = []
for r in rad_results:
    t = main_df.merge(r['df'], on="FileName", how="left")
    t['RadPrediction_Mapped'] = t['RadPrediction_Mapped'].fillna("Unknown")
    rec = recall_score(t['TrueCategory'], t['RadPrediction_Mapped'], labels=classes, average=None)
    all_recalls.append(rec)
avg_recall_rad = np.mean(all_recalls, axis=0)

per_class_data = []
for i, cls in enumerate(class_names_short):
    per_class_data.append({'Class': cls, 'Recall': recall_ai[i], 'Predictor': 'AI Model'})
    per_class_data.append({'Class': cls, 'Recall': recall_best[i], 'Predictor': f"Best Rad ({best_rad['id']})"})
    per_class_data.append({'Class': cls, 'Recall': avg_recall_rad[i], 'Predictor': 'Avg Radiologist'})

pc_df = pd.DataFrame(per_class_data)

plt.figure(figsize=(12, 6))
sns.barplot(x='Class', y='Recall', hue='Predictor', data=pc_df, palette=['#e74c3c', '#3498db', '#95a5a6'])
plt.title('Per-Class Sensitivity (Recall): AI vs Radiologists', fontsize=16, fontweight='bold')
plt.ylabel('Sensitivity', fontsize=12)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')

for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.2f')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_sensitivity.png", dpi=300)
plt.close()

print("Analysis Complete. Charts updated.")
