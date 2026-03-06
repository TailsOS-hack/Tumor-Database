# CLAUDE.md — AI Assistant Guide for Tumor-Database

## Project Overview

This is a **medical imaging AI system** for classifying brain MRI scans. It uses a hierarchical deep learning pipeline to detect and classify brain tumors (glioma, meningioma, pituitary) and Alzheimer's-related dementia stages (mild, moderate, very mild). A Tkinter-based GUI generates professional radiology-style PDF reports, optionally augmented by a local LLM (Ollama/llava:7b).

---

## Repository Structure

```
Tumor-Database/
├── src/                            # All Python source code
│   ├── radiology_report_gui.py     # Main GUI application (886 lines) — primary entry point
│   ├── train_complete_suite.py     # Master training script for all 3 models (394 lines)
│   ├── train_gatekeeper.py         # Standalone Gatekeeper (ResNet50) training
│   ├── train_alzheimers.py         # Standalone Dementia model (MobileNetV3) training
│   ├── gatekeeper_model.py         # ResNet50 architecture definition (35 lines)
│   └── data_loader.py              # Dataset loading and transforms (134 lines)
├── data/
│   ├── brain_tumor/
│   │   ├── Training/{glioma,meningioma,notumor,pituitary}/   # ~5,700 training images
│   │   └── Testing/{glioma,meningioma,notumor,pituitary}/    # ~1,300 test images
│   ├── alzheimers/
│   │   ├── MildDemented/
│   │   ├── ModerateDemented/
│   │   ├── NonDemented/
│   │   └── VeryMildDemented/                                  # ~44,000 total images
│   └── evaluation/
│       ├── images/                   # 100-image radiologist test set (anonymized)
│       ├── ground_truth/             # radiologist_test_key.csv (labels)
│       ├── model_results/            # model_predictions.csv
│       └── radiologist_results/      # Per-radiologist xlsx files
├── models/
│   ├── gatekeeper_classifier.pt      # ResNet50 router (~17MB, git-ignored)
│   ├── brain_tumor_classifier.pt     # EfficientNet-B3 (~43MB, git-ignored)
│   ├── alzheimers_classifier.pt      # MobileNetV3-Large (~17MB, git-ignored)
│   └── backup/                       # Model checkpoints
├── data_visualization/
│   ├── visualize_performance.py      # General performance metrics + charts
│   ├── evaluate_tumor_only.py        # Dedicated tumor model evaluation
│   └── compare_rad_vs_ai.py          # Radiologist vs AI comparison analysis
├── training_logs/                    # PNG plots + JSON history from training runs
├── Sample Reports/                   # Anonymized sample PDF outputs
├── .github/workflows/                # GitHub Actions (Gemini integration)
├── README.md                         # Installation and usage guide
├── GEMINI.md                         # Development log and architectural decisions
└── requirements.txt                  # Python dependencies
```

---

## Architecture: Hierarchical Classification Pipeline

The system uses a **3-stage routing pipeline** — never collapse this into a single model:

```
Input MRI Image (224×224 RGB)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Gatekeeper (ResNet50) │  3-way classifier
│  → class 0: Normal              │
│  → class 1: Tumor               │
│  → class 2: Dementia            │
└─────────────────────────────────┘
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────────────────────┐
│ Stage 2:     │  │ Stage 3:                     │
│ Tumor        │  │ Dementia Specialist           │
│ Specialist   │  │ (MobileNetV3-Large)           │
│ (EfficientNet│  │ → MildDemented               │
│  -B3)        │  │ → ModerateDemented           │
│ → Glioma     │  │ → VeryMildDemented           │
│ → Meningioma │  └──────────────────────────────┘
│ → Pituitary  │
└──────────────┘
```

**Key design decisions (do not change without strong justification):**
- The Gatekeeper unifies `notumor` (brain_tumor dataset) and `NonDemented` (Alzheimer's dataset) into a single **"Normal"** class (label 0). This avoids overlap between the two specialist domains.
- Each model is saved as a **complete PyTorch object** (`torch.save(model, path)`), not just state_dict. Load with `torch.load(path, map_location=device)`.
- Device priority: CUDA → DirectML (Windows) → CPU.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch + TorchVision |
| Model Architectures | ResNet50, EfficientNet-B3, MobileNetV3-Large |
| Image Processing | OpenCV, Pillow, torchvision.transforms |
| GUI | Tkinter + tkcalendar + tkhtmlview |
| PDF Generation | WeasyPrint (HTML/CSS → PDF) |
| LLM Integration | Ollama (`llava:7b`, local, optional) |
| Data Analysis | pandas, NumPy, scikit-learn, matplotlib, seaborn |
| Dataset Download | kagglehub |

---

## Running the Project

### Prerequisites

```bash
pip install -r requirements.txt
# Linux also needs: sudo apt install python3-tk libgtk-3-dev
# macOS: brew install gobject-introspection gtk+3
# Ollama (optional): https://ollama.com — pull llava:7b
```

### Main Application (GUI)

```bash
python -m src.radiology_report_gui
```

- GUI loads all 3 models from `models/` at startup
- Upload a brain MRI → classify → generate PDF report
- AI auto-detection tab uses `llava:7b` via Ollama for image understanding

### Training All Models

```bash
python src/train_complete_suite.py
```

Trains sequentially: Gatekeeper → Tumor Specialist → Dementia Specialist. Saves models to `models/` and plots to `training_logs/`.

Individual model training:
```bash
python src/train_gatekeeper.py
python src/train_alzheimers.py
```

### Evaluation and Visualization

```bash
python data_visualization/evaluate_tumor_only.py    # Tumor model on 100-image test set
python data_visualization/visualize_performance.py  # Training metrics and confusion matrices
python data_visualization/compare_rad_vs_ai.py      # Radiologist vs AI performance comparison
```

---

## Data Conventions

- **Image format:** JPG/PNG/JPEG, resized to **224×224** for all models
- **Normalization:** ImageNet statistics — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Train/val split:** 80/20 with `seed=42` — always use this seed for reproducibility
- **Data augmentation (training only):** random horizontal flip, rotation ±15°, color jitter; never augment validation/test sets
- **Class balance:** `data_loader.py` uses undersampling to handle class imbalance in Alzheimer's dataset

---

## Code Conventions

### Style
- Python 3.10+; type hints in function signatures where practical
- Descriptive variable names: `pred_label`, `confidence_str`, `self.gatekeeper_model`
- Use `logging` module for training progress (not bare `print`)
- `try/except` blocks around model I/O and GUI operations
- Keep GUI operations off the main thread for long-running tasks (use `threading`)

### PyTorch Patterns
```python
# Inference — always use this pattern
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Training — always call these
model.train()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Save complete model object (not state_dict)
torch.save(model, "models/my_model.pt")

# Load with device mapping
model = torch.load("models/my_model.pt", map_location=device)
```

### Model Architecture Pattern
Each model file follows this structure:
1. Architecture function: `build_<name>_model(num_classes, device) → nn.Module`
2. Transform function: `get_<name>_transform() → transforms.Compose`
3. Dataset class (if custom needed)
4. Training loop with epoch logging
5. Save model artifact

### File Naming
- Model files: `<name>_classifier.pt`
- Training plots: `training_logs/<name>_<metric>.png`
- Evaluation outputs: `data_visualization/comparison/<description>.png`

---

## Performance Benchmarks (as of Feb 2026)

| Model | Task | Accuracy |
|-------|------|----------|
| Gatekeeper (ResNet50) | Normal/Tumor/Dementia routing | ~80% |
| Tumor Specialist (EfficientNet-B3) | Glioma/Meningioma/Pituitary | ~80-83% |
| Dementia Specialist (MobileNetV3) | 4-stage Alzheimer's | ~80% |
| **Radiologist Average** | Brain tumor classification | **37-39.67%** |

The AI outperforms the 3-radiologist human baseline by ~40 percentage points on the 100-image evaluation set.

---

## Git Workflow

- **Main branch:** `master`
- **Feature/AI branches:** prefix `claude/` for Claude sessions, `gemini/` for Gemini sessions
- **Commit message conventions:**
  - `feat:` — new features
  - `refactor:` — code restructuring
  - `docs:` — documentation changes
  - `fix:` — bug fixes
  - `train:` / `eval:` — model training and evaluation work

### What is Git-Ignored
- `models/*.pt` — trained model files (too large; regenerate with training scripts)
- `models/Collab_Train/Train_Config/data.zip` — Colab training data archive

---

## Key Files for AI Assistants

When making changes, these are the most critical files to understand:

| Priority | File | Why |
|----------|------|-----|
| High | `src/radiology_report_gui.py` | Central application; model loading, inference pipeline, GUI |
| High | `src/train_complete_suite.py` | Authoritative training configuration for all models |
| High | `src/data_loader.py` | Dataset abstractions used by all training scripts |
| Medium | `src/gatekeeper_model.py` | Gatekeeper architecture definition |
| Medium | `data_visualization/compare_rad_vs_ai.py` | Evaluation methodology and metrics |
| Reference | `GEMINI.md` | Detailed development history and architectural rationale |
| Reference | `README.md` | Installation and platform-specific setup |

---

## Common Tasks and Pitfalls

### Adding a New Model
1. Create `src/train_<name>.py` following existing training script patterns
2. Register model loading in `src/radiology_report_gui.py` (model init section)
3. Add transform function to `src/data_loader.py` or new model file
4. Save to `models/<name>_classifier.pt`

### Modifying the Gatekeeper
- The unified "Normal" class (index 0) **must** remain the first class; downstream label mapping depends on this ordering
- Any change to class indices requires updating `radiology_report_gui.py` label maps

### Adding GUI Features
- Long operations (model inference, LLM calls) must run in a background thread; use `threading.Thread`
- PDF generation uses WeasyPrint — pass HTML strings, not file paths
- Calendar widget from `tkcalendar` (not standard Tkinter)

### Cross-Platform Notes
- GPU detection tries CUDA first, then `torch_directml` (Windows), then CPU — do not assume CUDA
- WeasyPrint requires GTK3 on Windows (separate installer); Linux needs `libgtk-3-dev`
- Ollama must be running locally before launching the GUI if using AI auto-detection

### Do Not
- Do not merge `notumor` and `NonDemented` into separate "Normal" classes — this breaks the Gatekeeper routing logic
- Do not use `state_dict` for saving models — the codebase loads complete model objects
- Do not run data augmentation on validation or test splits
- Do not hardcode device as `cuda` — always use the device detection pattern from existing code

---

## CI/CD

GitHub Actions workflows in `.github/workflows/` are for Gemini AI integration (triage, review, dispatch). There are no automated test or build pipelines currently. Evaluation is run manually via the scripts in `data_visualization/`.
