# Hybrid AI Brain Tumor Radiology Report Generator

This project uses a sophisticated hybrid AI system to classify brain tumors from MRI images and generate detailed, radiologist-style reports. It combines a fine-tuned PyTorch model for accurate classification with a local multimodal Large Language Model (LLM) for advanced visual analysis and report drafting.

![GUI Screenshot](<placeholder_for_advanced_gui_screenshot.png>)
*(Note: You can replace the placeholder above with a screenshot of the `src/radiology_report_gui.py` application in action.)*

## Project Structure
```
.
├── data
│   └── brain-tumor-mri-dataset
│       ├── Testing
│       └── Training
├── models
│   └── brain_tumor_classifier.pt
├── src
│   ├── __init__.py
│   ├── radiology_report_gui.py (Main Application)
│   └── train.py
├── .gitattributes
├── README.md
└── requirements.txt
```

## Features

-   **Hybrid AI Analysis:**
    -   **Stage 1 (Classification):** Utilizes a fine-tuned `EfficientNet-B3` model in PyTorch to achieve high-accuracy classification among four classes: `glioma`, `meningioma`, `no tumor`, and `pituitary`.
    -   **Stage 2 (Visual Analysis):** Employs a local multimodal LLM (`ollama/llava:7b`) to "see" the MRI scan. Based on the initial classification, it describes the tumor's specific visual characteristics and estimates its size.
-   **Advanced Report Generation:** The `llava` model drafts a detailed radiology report, with conditional logic to handle "no tumor" cases gracefully.
-   **Embedded Imagery:** The final PDF report dynamically embeds the analyzed MRI scan at the bottom for complete reference.
-   **PDF Encryption:** Generated reports can be encrypted with a user-provided password for enhanced security of sensitive patient data.
-   **Rich GUI:** A user-friendly interface for entering patient data, selecting an image, and generating/saving the report.
-   **GPU Accelerated:** Supports both GPU (CUDA/DirectML) and CPU for model inference.

## Requirements

### 1. Ollama Setup
This project requires a local Ollama instance to be running.

1.  **Install Ollama:** Follow the instructions at [ollama.com](https://ollama.com/).
2.  **Pull the Multimodal Model:** Once Ollama is running, open a terminal and pull the `llava:7b` model. This is required for the visual analysis part of the report generation.
    ```bash
    ollama pull llava:7b
    ```

### 2. Python Dependencies
All Python dependencies are listed in `requirements.txt`. This includes `torch`, `ollama`, `pypdf`, `WeasyPrint`, `markdown`, and `tkcalendar`.

Install them using pip:
```bash
pip install -r requirements.txt
```

### 3. System-Level Dependencies (for Windows)
The PDF export feature relies on the `WeasyPrint` library, which requires a one-time installation of the GTK+ runtime on Windows.

1.  **Install MSYS2:** Download and run the installer from [msys2.org](https://www.msys2.org/).
2.  **Install GTK3 via MSYS2:** Open the MSYS2 terminal and run `pacman -S mingw-w64-x86_64-gtk3`.
3.  **Update PATH:** Add `C:\msys64\mingw64\bin` to your system's PATH environment variable to make the GTK+ libraries discoverable.

## Usage

### 1. Train the Classifier (Optional)

If you want to train the PyTorch classifier from scratch, run the training script. This will save the best-performing model to `models/brain_tumor_classifier.pt`.

```bash
python src/train.py
```

### 2. Run the Report Generation GUI

This is the main application for generating reports. **Ensure your local Ollama application is running before you start.**

```bash
python src/radiology_report_gui.py
```

The application will automatically load the trained classifier if it exists at the default path.

**How to use the GUI:**
1.  Fill in the **Patient Information** (Name, Patient ID, and DOB).
2.  If the classifier model doesn't load automatically, click **"Load Classifier Model"** and select the `.pt` file.
3.  Click **"Choose Image..."** to select an MRI scan.
4.  Click **"Analyze & Generate Report"**. The application will:
    -   Classify the image using the PyTorch model.
    -   Pass the image and classification to the `llava` model to write a detailed report, including a description and size estimate.
5.  Once the report is generated, click **"Save as PDF"**. You will be prompted to add an optional password to encrypt the report.