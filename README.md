# AI-Powered Neuro-Radiology Report Generator

This project is an advanced medical imaging analysis tool. It uses deep learning to analyze MRI scans for both **Brain Tumors** and **Alzheimer's/Dementia** signs, then generates grounded draft reports from structured classifier evidence. The reporting path is deterministic by default so it does not ask an LLM to invent lesion measurements, locations, mass effect, atrophy measurements, or other unsupported clinical findings.

## Features

-   **Hierarchical Classification:** Uses a multi-stage AI pipeline for maximum accuracy:
    -   **Gatekeeper Model:** A `ResNet50` 3-way classifier that first determines if an MRI is **Normal**, **Tumor**, or **Dementia**.
    -   **Specialized Classifiers:**
        -   **Brain Tumor Classifier:** `EfficientNet-B3` (PyTorch) for specific tumor types (Glioma, Meningioma, Pituitary).
        -   **Alzheimer's Classifier:** `MobileNetV3-Large` (PyTorch) for dementia stages (Mild, Moderate, Very Mild).
    -   **Unified Normal Class:** The system intelligently identifies healthy scans from both datasets as a single "Normal" category.
-   **Grounded Report Maker:** Generates draft reports from the classifier label, confidence, user-entered exam details, and model validation context. Unsupported findings are explicitly marked as not assessed instead of being hallucinated.
-   **Optional AI Metadata Assist:** The GUI can use a multimodal AI model only for basic acquisition metadata. The prompt refuses diagnosis/pathology inference and falls back to manual entry when uncertain.
-   **Standardized Reporting:** Generates reports with a strict evidence schema for consistency across Findings, Impression, Technique, classifier evidence, and safety limitations.
-   **PDF Export:** Saves draft reports as PDF documents with the MRI image embedded, timestamped footer, page numbering, and optional password encryption.
-   **User-Friendly GUI:** A modern Tkinter interface with tabs for Patient Info and Exam Details.
-   **Data Visualization:** Includes a suite to benchmark model performance and generate accuracy heatmaps.

## Project Structure

-   `src/radiology_report_gui.py`: The main application.
-   `src/train_complete_suite.py`: The master script to train all 3 models sequentially.
-   `src/gatekeeper_model.py`: Definition of the routing model.
-   `data_visualization/`:
    -   `visualize_performance.py`: General model benchmark tools.
    -   `compare_rad_vs_ai.py`: Script to compare AI accuracy vs human radiologist.
    -   `evaluate_tumor_only.py`: Dedicated evaluation script for the Brain Tumor classifier.
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
- **Ollama** (optional, only for metadata assist; grounded report generation does not require an LLM)

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

3.  **Optional: Install & Setup Ollama**
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

3.  **Optional: Install & Setup Ollama**
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

4.  **Optional: Install & Setup Ollama**
    -   Download the Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).
    -   Run the installer.
    -   Open PowerShell or Command Prompt and run:
        ```bash
        ollama pull llava:7b
        ```

## How to Run

### Clean Research Training Flow
For strict tumor-vs-dementia retraining, hierarchical evaluation, the single 8-class baseline, and Colab multimodal/LoRA work, use the execution guide:

```bash
python -m src.experiment_pipeline create-manifest
python -m src.experiment_pipeline train --task binary --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline train --task tumor --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline train --task dementia --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline evaluate-hierarchical --split test
python -m src.experiment_pipeline train --task eight_class --epochs 30 --batch-size 32 --pretrained
```

See `docs/ML_EXECUTION_FLOW.md` and `docs/PUBLICATION_NOTES.md` for the full task order, leakage controls, Colab plan, and publication tables. See `docs/GROUNDED_REPORTING.md` for the non-hallucinating report-generation path and current specialist accuracy table.

For cloud execution, dispatch the **ML Experiment Suite** GitHub Actions workflow in `smoke` mode for validation or `full` mode on a CUDA/GPU runner. For a single Google Colab run that trains the full suite and downloads a model/metrics bundle, use `docs/COLAB_FULL_TRAINING.md`. For multimodal LLM benchmarking and LoRA fine-tuning, open `notebooks/multimodal_llm_lora_colab.ipynb` in Google Colab.

### Main GUI Application
Run the main GUI application (using the module flag to ensure imports work correctly):

```bash
python -m src.radiology_report_gui
```

1.  **Patient Info:** Enter Patient Name, ID, and Date of Birth.
2.  **Scan:** Click **"Scan"** to load an MRI image.
3.  **Exam Details:** Switch to the "Exam Details" tab.
    -   Click **"AI Auto-Detect"** to let the optional AI assistant fill basic acquisition metadata only.
    -   Or click **"Manual Entry"** to fill the details yourself.
4.  **Analyze:** Once details are confirmed, click **"Analyze & Generate Report"**.
5.  **Export:** Review the generated report and click **"Save as PDF"**.

### Grounded Report CLI

Generate a deterministic report from a classifier result:

```bash
python -m src.grounded_report --prediction glioma --confidence 97.2% --model-used "Brain Tumor Model" --format markdown
```

Or use hierarchical inference output:

```bash
python -m src.hierarchical_inference path/to/mri.png > prediction.json
python -m src.grounded_report --prediction-json prediction.json --format html --output grounded_report.html
```

### Performance Visualization
To generate performance graphs and heatmaps for the models:

```bash
python data_visualization/visualize_performance.py
```
This will run a subset of the data through both models and generate `.png` plots in the `data_visualization` folder showing accuracy and confusion matrices.

To run the dedicated Brain Tumor model evaluation (producing detailed metrics and ROC curves):
```bash
python data_visualization/evaluate_tumor_only.py
```
