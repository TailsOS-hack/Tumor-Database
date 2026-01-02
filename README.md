# AI-Powered Neuro-Radiology Report Generator

This project is an advanced medical imaging analysis tool. It uses deep learning to analyze MRI scans for both **Brain Tumors** and **Alzheimer's/Dementia** signs, and then employs a local Large Language Model (LLM) to generate detailed, professional radiology reports.

## Features

-   **Hierarchical Classification:** Uses a multi-stage AI pipeline for maximum accuracy:
    -   **Gatekeeper Model:** A `ResNet50` 3-way classifier that first determines if an MRI is **Normal**, **Tumor**, or **Dementia**.
    -   **Specialized Classifiers:**
        -   **Brain Tumor Classifier:** `EfficientNet-B3` (PyTorch) for specific tumor types (Glioma, Meningioma, Pituitary).
        -   **Alzheimer's Classifier:** `MobileNetV3-Large` (PyTorch) for dementia stages (Mild, Moderate, Very Mild).
    -   **Unified Normal Class:** The system intelligently identifies healthy scans from both datasets as a single "Normal" category, generating immediate reports without unnecessary processing.
-   **Generative AI Reporting:** Uses a local Multimodal LLM (`ollava/llava:7b` via Ollama) to "see" the image and draft a full radiology report.
-   **PDF Export:** Saves reports as professional PDF documents with the MRI image embedded and optional password encryption.
-   **User-Friendly GUI:** A modern Tkinter interface for easy patient data entry and analysis.
-   **Data Visualization:** Includes a suite to benchmark model performance and generate accuracy heatmaps.

## Project Structure

-   `src/radiology_report_gui.py`: The main application.
-   `src/train_complete_suite.py`: The master script to train all 3 models sequentially.
-   `src/gatekeeper_model.py`: Definition of the routing model.
-   `data_visualization/`: Performance analysis tools.
-   `models/`:
    -   `gatekeeper_classifier.pt`: ResNet50 router model.
    -   `brain_tumor_classifier.pt`: EfficientNet-B3 model.
    -   `alzheimers_classifier.pt`: MobileNetV3 model.
-   `data/`: Directory for datasets (Brain Tumor and Alzheimer's).

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
Run the main GUI application:

```bash
python -m src.radiology_report_gui
```

1.  Enter Patient Name, ID, and Date of Birth.
2.  Click **"Choose Image..."** to load an MRI scan (Tumor or Alzheimer's).
3.  Click **"Analyze & Generate Report"**.
4.  The system will classify the image and generate a report.
5.  Click **"Save as PDF"** to export.

### Performance Visualization
To generate performance graphs and heatmaps for the models:

```bash
python data_visualization/visualize_performance.py
```
This will run a subset of the data through both models and generate `.png` plots in the `data_visualization` folder showing accuracy and confusion matrices.