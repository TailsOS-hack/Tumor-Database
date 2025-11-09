# AI-Powered Brain Tumor Radiology Report Generator

This project uses a deep learning model (PyTorch) to classify brain tumors from MRI images and leverages a local Large Language Model (LLM) via Ollama to generate radiologist-style reports.

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
│   ├── gui.py (Original simple classifier)
│   ├── radiology_report_gui.py (Advanced report generator)
│   └── train.py
├── .gitattributes
├── README.md
└── requirements.txt
```

## Features

-   **AI Classification:** Utilizes a pre-trained `EfficientNet-B3` model to distinguish between four classes: `glioma`, `meningioma`, `no tumor`, and `pituitary`.
-   **LLM Report Generation:** Integrates with a local LLM (Ollama) to draft a detailed radiology report based on the AI's findings.
-   **Rich GUI:** An advanced GUI that allows for structured patient data entry (Name, DOB, Patient ID), image selection, report generation, and PDF export.
-   **Markdown & PDF Export:** Renders the generated report from Markdown and allows for easy exporting to a PDF document.
-   **GPU Accelerated:** The training script and inference GUI can be run on a GPU (CUDA/DirectML) or CPU.

## Requirements

### Python Dependencies
All Python dependencies are listed in the `requirements.txt` file. This includes `torch`, `ollama`, `WeasyPrint`, `markdown`, and `tkcalendar`.

Install them using pip:
```bash
pip install -r requirements.txt
```

### System-Level Dependencies (for Windows)
The PDF export feature relies on the `WeasyPrint` library, which requires a one-time installation of the GTK+ runtime on Windows.

1.  **Install MSYS2:** Download and run the installer from [msys2.org](https://www.msys2.org/).
2.  **Install GTK3 via MSYS2:** Open the MSYS2 terminal and run `pacman -S mingw-w64-x86_64-gtk3`.
3.  **Update PATH:** Add `C:\msys64\mingw64\bin` to your system's PATH environment variable to make the GTK+ libraries discoverable.

## Usage

### 1. Train the Model (Optional)

If you want to train the classifier from scratch, run the training script. This will save the best-performing model to `models/brain_tumor_classifier.pt`.

```bash
python src/train.py
```

### 2. Run the Report Generation GUI

This is the main application for generating reports.

```bash
python src/radiology_report_gui.py
```

The application will automatically load the trained model if it exists at the default path.

**How to use the GUI:**
1.  Fill in the **Patient Information** (Name, Patient ID, and DOB) at the top left.
2.  If the model doesn't load automatically, click **"Load Classifier Model"** and select the `.pt` file.
3.  Click **"Choose Image..."** to select an MRI scan.
4.  Click **"Analyze & Generate Report"**. The application will classify the image and use the local LLM to write a report.
5.  Once the report is generated, click **"Save as PDF"** to export the report.