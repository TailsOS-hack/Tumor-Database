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

## Installation Guide

### Prerequisites
- **Python 3.10+**
- **Ollama** (for local AI report generation)

### 🐧 Linux (Ubuntu/Debian)

1.  **Install System Dependencies**
    The GUI requires Tkinter, and the PDF generator (`WeasyPrint`) needs specific system libraries.
    ```bash
    sudo apt update
    sudo apt install python3-tk libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt --break-system-packages
    ```
    *(Note: Using a virtual environment `venv` is recommended if you prefer not to install system-wide.)*

3.  **Install & Setup Ollama**
    -   Install Ollama:
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ```
    -   Start the Ollama server:
        ```bash
        ollama serve
        ```
    -   In a new terminal, pull the required multimodal model:
        ```bash
        ollama pull llava:7b
        ```

### 🍎 macOS

1.  **Install System Dependencies (via Homebrew)**
    You need Homebrew installed first.
    ```bash
    brew install python-tk pango libffi
    ```

2.  **Install Python Dependencies**
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Install & Setup Ollama**
    -   Download and install Ollama from [ollama.com/download/mac](https://ollama.com/download/mac).
    -   Open the Ollama application.
    -   Run the following in your terminal to download the model:
        ```bash
        ollama pull llava:7b
        ```

### 🪟 Windows

1.  **Install Python**
    Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/). Ensure you check **"Add Python to PATH"** during installation.

2.  **Install GTK3 Runtime (Crucial for PDF Export)**
    The PDF generation library (`WeasyPrint`) requires the GTK3 runtime.
    -   Download and install the [GTK3 Installer for Windows](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
    -   *Alternative (via MSYS2):*
        1.  Install MSYS2 from [msys2.org](https://www.msys2.org/).
        2.  Run `pacman -S mingw-w64-x86_64-gtk3` in the MSYS2 terminal.
        3.  Add `C:\msys64\mingw64\bin` to your System PATH environment variable.

3.  **Install Python Dependencies**
    Open Command Prompt (cmd) or PowerShell and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install & Setup Ollama**
    -   Download the Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).
    -   Run the installer.
    -   Open PowerShell or Command Prompt and run:
        ```bash
        ollama pull llava:7b
        ```

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
