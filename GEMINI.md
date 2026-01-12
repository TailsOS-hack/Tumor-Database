# Gemini Project Context: AI-Powered Brain Tumor Radiology Report Generator

## Project Overview

This project is an advanced medical imaging analysis tool. It combines a deep learning model (PyTorch) for classifying brain tumors from MRI scans with a local Large Language Model (LLM) via Ollama to automatically generate detailed, radiologist-style reports.

The project is organized into the following structure:
- `src/`: Contains all Python source code.
  - `train_complete_suite.py`: The master script to train all 3 models (Gatekeeper, Tumor, Dementia).
  - `radiology_report_gui.py`: The primary GUI application for classification, report generation, and PDF export.
  - `gatekeeper_model.py`: Definition of the Gatekeeper architecture.
- `data/`: Contains the image dataset.
- `models/`: Contains the trained model checkpoints.
- `requirements.txt`: Lists all Python dependencies.

## Key Files

-   `src/radiology_report_gui.py`: The main GUI application. This `tkinter`-based GUI provides a complete workflow for loading the classifier, inputting patient data, analyzing an MRI image, generating a formatted report using a local LLM, and exporting the final report as a PDF.
-   `src/train_complete_suite.py`: A unified training script that trains the Gatekeeper, Tumor, and Dementia models sequentially.
-   `models/gatekeeper_classifier.pt`: The 3-way ResNet50 classifier (Normal vs. Tumor vs. Dementia).
-   `models/brain_tumor_classifier.pt`: The specialized EfficientNet-B3 model for tumor types (Glioma, Meningioma, Pituitary).
-   `models/alzheimers_classifier.pt`: The specialized MobileNetV3 model for dementia stages (Mild, Moderate, Very Mild).

## Setup and Dependencies

### Python Dependencies
This project is written in Python. All dependencies, including `torch`, `ollama`, `WeasyPrint`, `markdown`, and `tkcalendar`, are listed in `requirements.txt`.

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

### System-Level Dependencies (for Windows)
The PDF export feature relies on the `WeasyPrint` library, which requires a **one-time installation of the GTK+ runtime** on Windows.

1.  **Install MSYS2:** Download and run the installer from [msys2.org](https://www.msys2.org/).
2.  **Install GTK3 via MSYS2:** Open the MSYS2 terminal and run `pacman -S mingw-w64-x86_64-gtk3`.
3.  **Update PATH:** Add `C:\msys64\mingw64\bin` to your system's PATH environment variable. A restart of the terminal or system may be required.

## How to Run

### 1. Train the Models (Unified Suite)

To train all models from scratch, run the complete suite. This handles the Gatekeeper (3-way), Tumor, and Dementia models.

```bash
python src/train_complete_suite.py
```

### 2. Run the Report Generation GUI

After training is complete, launch the main GUI application.

```bash
python src/radiology_report_gui.py
```

**GUI Workflow:**
1.  Enter the patient's information (Name, DOB, Patient ID).
2.  The application auto-loads the models.
3.  Click **"Choose Image..."** to select an MRI scan.
4.  Click **"Analyze & Generate Report"**.
    -   **Stage 1:** Gatekeeper determines if the image is **Normal**, **Tumor**, or **Dementia**.
    -   **Stage 2:** If not Normal, the specific specialized model analyzes the subtype/stage.
    -   **Stage 3:** The local multimodal LLM generates a detailed report.
5.  Review the generated report.
6.  Click **"Save as PDF"** to export.

## Development Conventions

-   **Hierarchical Classification:** The system uses a Gatekeeper model to route images to the correct domain, preventing cross-domain errors.
-   **Unified "Normal" Class:** The Gatekeeper is trained to recognize "Normal" images from both the Tumor (no_tumor) and Alzheimer's (NonDemented) datasets as a single class.
-   **Model Checkpoints:** Models are saved in `models/` as self-contained files or state dicts.

## Development Log

### Session Summary: Training Completion & Verification

Following the architectural refactoring, the unified training suite was executed.

**Status:**
-   **Training Complete:** The `src/train_complete_suite.py` script has successfully generated all three required model files:
    -   `gatekeeper_classifier.pt` (3-way routing)
    -   `brain_tumor_classifier.pt` (Tumor specialists)
    -   `alzheimers_classifier.pt` (Dementia specialists)
-   **System Ready:** The `radiology_report_gui.py` is now fully operational with the new models and the "Unified Normal" classification logic.

### Session Summary: Radiologist Test Set Creation

Created a randomized test dataset for radiologist evaluation as requested.

**Key Actions:**
-   **Dataset Creation:** Generated `Test Images/` containing 100 images:
    -   20 images randomly selected from each of the 4 dementia categories (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`).
    -   20 additional random images from the remaining pool.
-   **Anonymization:** Files were renamed sequentially (`image_001.jpg` to `image_100.jpg`) and shuffled to blind the reviewers.
-   **Documentation:**
    -   `Test Images/score_sheet.csv`: A template for radiologists to record their diagnoses.
    -   `Test Image Answer Key/radiologist_test_key.csv`: A master key mapping filenames to their original ground-truth categories (moved to a separate folder for security).

### Session Summary: Unified Normal Class & 3-Way Gatekeeper

This session significantly refactored the classification architecture to improve robustness and simplify the "Normal" case handling.

**Key Changes:**
-   **Unified "Normal" Class:** Combined `no_tumor` (Brain Tumor dataset) and `NonDemented` (Alzheimer's dataset) into a single "Normal" class for the Gatekeeper.
-   **3-Way Gatekeeper:** Updated the Gatekeeper model (`src/gatekeeper_model.py`) to classify images into 3 categories: **Normal**, **Tumor**, and **Dementia**. This replaces the previous binary (Tumor vs. Dementia) approach.
-   **Unified Training Script:** Created `src/train_complete_suite.py` to train all three models (Gatekeeper, Tumor, Dementia) in a single workflow. This ensures data consistency (e.g., excluding "Normal" images from the specialized training sets).
-   **GUI Updates:**
    -   Updated `src/radiology_report_gui.py` to handle the 3-way Gatekeeper output.
    -   Added logic to immediately generate a "Normal" report if the Gatekeeper predicts Class 0, bypassing the specialized models.
    -   Standardized all image transforms to **224x224** resolution across the entire pipeline.
    -   Removed `NonDemented` from the Alzheimer's specialized model logic.

### Session Summary: Final Polishing and Formatting (Previous)
... (Previous logs retained)