# Gemini Project Context: AI-Powered Brain Tumor Radiology Report Generator

## Project Overview

This project is an advanced medical imaging analysis tool. It combines a deep learning model (PyTorch) for classifying brain tumors from MRI scans with a local Large Language Model (LLM) via Ollama to automatically generate detailed, radiologist-style reports.

The project is organized into the following structure:
- `src/`: Contains all Python source code.
  - `train.py`: A command-line script to train the `EfficientNet-B3` model.
  - `radiology_report_gui.py`: The primary GUI application for classification, report generation, and PDF export.
  - `gui.py`: The original, simpler GUI for classification only.
- `data/`: Contains the image dataset.
- `models/`: Contains the trained model checkpoints.
- `requirements.txt`: Lists all Python dependencies.

## Key Files

-   `src/radiology_report_gui.py`: The main application. This `tkinter`-based GUI provides a complete workflow for loading the classifier, inputting patient data, analyzing an MRI image, generating a formatted report using a local LLM, and exporting the final report as a PDF.
-   `src/train.py`: The script for training the image classifier. It handles data loading, augmentation, model training, validation, and saving the best model.
-   `src/gui.py`: The original, basic GUI for classifying images. Superseded by `radiology_report_gui.py`.
-   `models/brain_tumor_classifier.pt`: The trained PyTorch model checkpoint saved by the training script.
-   `requirements.txt`: The file listing project dependencies for easy setup.

## Setup and Dependencies

### Python Dependencies
This project is written in Python. All dependencies, including `torch`, `ollama`, `WeasyPrint`, and `markdown`, are listed in `requirements.txt`.

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

### 1. Train the Model (Optional)

To train the classifier from scratch, run the training script. The script will use the images in `data/brain-tumor-mri-dataset/` and save the best-performing model to `models/brain_tumor_classifier.pt`.

```bash
python src/train.py
```

### 2. Run the Report Generation GUI

After a model has been trained and saved, you can launch the main GUI application.

```bash
python src/radiology_report_gui.py
```

**GUI Workflow:**
1.  Enter the patient's information in the provided text box.
2.  The application will attempt to auto-load the model from `models/brain_tumor_classifier.pt`. If needed, load it manually via the **"Load Classifier Model"** button.
3.  Click **"Choose Image..."** to select an MRI scan.
4.  Click **"Analyze & Generate Report"**. This performs the classification and calls the local Ollama LLM to draft the report.
5.  Review the generated report.
6.  Click **"Save as PDF"** to export the final report to a PDF file.

## Development Conventions

-   **Separation of Concerns:** The project separates the ML model training (`train.py`), the core classification logic, and the user interface (`radiology_report_gui.py`). It further separates the classification task (PyTorch model) from the text generation task (Ollama LLM).
-   **Configuration:** The training script uses `argparse` for hyperparameters. The GUI is self-contained.
-   **Model Checkpoints:** The training script saves self-contained model files that include metadata like class names and architecture.
-   **Device Agnostic:** The code attempts to use available hardware acceleration (CUDA or DirectML) and falls back to the CPU.