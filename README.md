# AI-Powered Neuro-Radiology Report Generator

This project is an advanced medical imaging analysis tool. It uses deep learning to analyze MRI scans for both **Brain Tumors** and **Alzheimer's/Dementia** signs, and employs a local Large Language Model (LLM) via Ollama to automatically generate detailed, professional radiology reports following strict clinical templates.

## Features

-   **Hierarchical Classification:** Uses a multi-stage AI pipeline for maximum accuracy:
    -   **Gatekeeper Model:** A `ResNet50` 3-way classifier that first determines if an MRI is **Normal**, **Tumor**, or **Dementia**.
    -   **Specialized Classifiers:**
        -   **Brain Tumor Classifier:** `EfficientNet-B3` (PyTorch) for specific tumor types (Glioma, Meningioma, Pituitary).
        -   **Alzheimer's Classifier:** `MobileNetV3-Large` (PyTorch) for dementia stages (Mild, Moderate, Very Mild).
    -   **Unified Normal Class:** The system intelligently identifies healthy scans from both datasets as a single "Normal" category.
-   **AI Auto-Detect Metadata:** The system uses a multimodal AI model to analyze the scan image and automatically infer technical details (e.g., "Axial T2-weighted", "Contrast/Non-contrast") and the clinical reason for the exam.
-   **Standardized Reporting:** Generates reports using a strict JSON template to ensure professional consistency (Findings, Impression, Technique, etc.), avoiding common AI formatting errors.
-   **PDF Export:** Saves reports as professional "Final Report" PDF documents with the MRI image embedded, timestamped footer, page numbering, and optional password encryption.
-   **User-Friendly GUI:** A modern Tkinter interface with tabs for Patient Info and Exam Details.
-   **Data Visualization:** Includes a suite to benchmark model performance and generate accuracy heatmaps.

## Project Structure

-   `src/radiology_report_gui.py`: The main application.
-   `src/train_complete_suite.py`: The master script to train all 3 models sequentially.
-   `src/gatekeeper_model.py`: Definition of the routing model.
-   `data_visualization/`:
    -   `visualize_performance.py`: General model benchmark tools.
    -   `compare_rad_vs_ai.py`: Script to compare AI accuracy vs human radiologist.
-   `models/`:
    -   `gatekeeper_classifier.pt`: ResNet50 router model.
    -   `brain_tumor_classifier.pt`: EfficientNet-B3 model.
    -   `alzheimers_classifier.pt`: MobileNetV3 model.
-   `data/`:
    -   `evaluation/`: Contains test images, ground truth keys, and radiologist/AI results.
    -   `alzheimers/`: Training data for dementia.
    -   `brain_tumor/`: Training data for tumors.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install Ollama:**
    Ensure [Ollama](https://ollama.com/) is installed and the `llava:7b` model is pulled:
    ```bash
    ollama pull llava:7b
    ```
3.  **GTK3 (Windows only):**
    For PDF generation, you may need GTK3 installed via MSYS2 (see `WeasyPrint` docs).

## How to Run

### Main GUI Application
Run the main GUI application (using the module flag to ensure imports work correctly):

```bash
python -m src.radiology_report_gui
```

1.  **Patient Info:** Enter Patient Name, ID, and Date of Birth.
2.  **Scan:** Click **"Scan"** to load an MRI image.
3.  **Exam Details:** Switch to the "Exam Details" tab.
    -   Click **"AI Auto-Detect"** to let the AI guess the technique and reason from the image.
    -   Or click **"Manual Entry"** to fill the details yourself.
4.  **Analyze:** Once details are confirmed, click **"Analyze & Generate Report"**.
5.  **Export:** Review the generated report and click **"Save as PDF"**.

### Performance Visualization
To generate performance graphs and heatmaps for the models:

```bash
python data_visualization/visualize_performance.py
```
This will run a subset of the data through both models and generate `.png` plots in the `data_visualization` folder showing accuracy and confusion matrices.
