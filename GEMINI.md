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

-   `src/radiology_report_gui.py`: The main GUI application. This `tkinter`-based GUI provides a complete workflow for loading the classifier, inputting patient data, analyzing an MRI image, generating a formatted report using a local LLM, and exporting the final report as a PDF.
-   `src/train.py`: The script for training the image classifier. It handles data loading, augmentation, model training, validation, and saving the best model.
-   `src/gui.py`: The original, basic GUI for classifying images. Superseded by `radiology_report_gui.py`.
-   `models/brain_tumor_classifier.pt`: The trained PyTorch model checkpoint saved by the training script.
-   `requirements.txt`: The file listing project dependencies for easy setup.

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
1.  Enter the patient's information (Name, DOB, Patient ID) in the provided fields.
2.  The application will attempt to auto-load the model from `models/brain_tumor_classifier.pt`. If needed, load it manually via the **"Load Classifier Model"** button.
3.  Click **"Choose Image..."** to select an MRI scan.
4.  Click **"Analyze & Generate Report"**. This performs the classification and calls the local Ollama LLM to draft the report.
5.  Review the generated report.
6.  Click **"Save as PDF"** to export the final report to a PDF file.

## Development Conventions

-   **Separation of Concerns:** The project separates the ML model training (`train.py`), the core classification logic, and the user interface (`radiology_report_gui.py`). It further separates the classification task (PyTorch model) from the text generation task (Ollama LLM).
-   **Configuration:** The training script uses `argparse` for hyperparameters. The GUIs are self-contained.
-   **Model Checkpoints:** The training script saves self-contained model files that include metadata like class names and architecture.
-   **Device Agnostic:** The code attempts to use available hardware acceleration (CUDA or DirectML) and falls back to the CPU.

## Development Log

### Session Summary: Final Polishing and Formatting

This session focused on the final, detailed formatting of the PDF report to match the user's exact specifications. The process involved several iterations of refining the LLM prompt and the data passed to it, as the LLM's interpretation of formatting instructions proved to be inconsistent.

**Key Formatting Fixes:**

-   **Centered Title:** The main "RADIOLOGY REPORT" title is now reliably centered by wrapping the `<h1>` tag in a `<div style="text-align: center;">`.
-   **Patient Details:** To enforce new lines for each item, the patient information is now pre-formatted with `<br>` tags and bolded labels (`**Name:**`, `**DOB:**`) before being passed to the LLM prompt. This proved more reliable than asking the LLM to format the data itself.
-   **Bulleted Findings:** The prompt template was modified to explicitly include the hyphen (`-`) and a newline for each point in the "FINDINGS" section, forcing the LLM to generate a correctly formatted bulleted list.
-   **Confidence Score Display:** A bug where the confidence score was not being displayed (showing as `{confidence:.1f}%`) was fixed by pre-formatting the confidence value into a string in Python before inserting it into the prompt.
-   **Robust Image Embedding:** The most significant change was decoupling the image from the LLM's response. The final, robust solution involves programmatically appending the image's HTML block to the report's HTML *after* the LLM has generated the text, ensuring the image is always included in the PDF without relying on a fragile placeholder.

### Session Summary: Hybrid AI and Multimodal Analysis

This session marks a major architectural evolution of the project, moving from a text-based LLM approach to a sophisticated hybrid AI system. The core changes were driven by the need for the AI to "see" the MRI scan to provide more accurate descriptions and size estimations.

**Key Architectural Changes:**

-   **Hybrid AI Model:** The system now uses a two-stage process:
    1.  **Classification:** The fine-tuned PyTorch `EfficientNet-B3` model is retained for its primary, high-accuracy classification task (identifying the tumor type).
    2.  **Visual Analysis & Generation:** A local multimodal LLM (`ollava/llava:7b`) is introduced to perform visual analysis. It receives the MRI scan and the classification result from the first stage, and is tasked with describing the tumor's specific appearance and estimating its size.

-   **Component Decoupling:** The original `tumor_size_analyzer.py` script, which relied on OpenCV and fixed conversion ratios, has been removed. The responsibility for size estimation is now delegated to the multimodal LLM, streamlining the pipeline.

**New Features & Major Refinements:**

-   **Multimodal Report Generation:** The LLM prompt system was completely overhauled. The application now sends the image data directly to the `llava` model, instructing it to act as a radiologist, accept the classifier's finding, and generate a detailed report based on what it "sees".
-   **PDF Encryption:** To address the sensitive nature of medical data, a new feature was added to encrypt the exported PDF report with a user-provided password, using the `pypdf` library.
-   **Dynamic "No Tumor" Reports:** The prompt system now has conditional logic to generate a much simpler and more appropriate report when the classifier identifies an image as having "no tumor".
-   **Improved Report Layout:**
    -   The patient details are now formatted into a robust HTML table to ensure correct line breaks and alignment in the final PDF.
    -   The MRI scan image is now embedded at the bottom of the report under the "IMPRESSION" section and is centered on the page.

### Session Summary

This session focused on iteratively refining the GUI and report generation functionality of the `radiology_report_gui.py` application based on user feedback.

**Key Changes:**

-   **GUI Overhaul:** The patient information input was completely redesigned. The single text box was replaced with three distinct fields: "Name" (text entry), "Patient ID" (text entry), and "DOB" (a user-friendly date picker). This improves data entry accuracy and user experience.
-   **Report Generation Refinement:** The LLM prompt was significantly improved to produce more professional and clinically relevant reports. The "History" and "Technique" sections were removed for conciseness, and the "Patient Details" are now formatted with clear line breaks. The prompt also now explicitly prevents the LLM from redacting patient information.
-   **Performance Tuning:** The application's responsiveness was improved by reverting the full background processing for classification. The initial (and fast) classification now runs in the main thread to provide immediate feedback to the user, while the slower LLM report generation remains in a background thread to prevent the GUI from freezing.
-   Added `Flask` to `requirements.txt` to support the web application.

### Session Summary: Alzheimer's Detection & Dual-Model Integration

This session expanded the project's scope from a dedicated Brain Tumor classifier to a multi-purpose Neuro-Radiology tool by integrating Alzheimer's/Dementia detection.

**Key Achievements:**
-   **New Dataset & Model:**
    -   Downloaded the "Alzheimer's Dataset (4 class of Images)" from Kaggle.
    -   Trained a `MobileNetV3-Large` model on this dataset for 5 epochs.
    -   Achieved a validation accuracy of **90.20%**.
    -   Saved the model as `models/alzheimers_classifier.pt`.
-   **Dual-Model Architecture:**
    -   Updated `src/radiology_report_gui.py` to load *both* the original Brain Tumor model (`EfficientNet-B3`) and the new Alzheimer's model.
    -   Implemented a **Competitive Classification Logic**: When an image is uploaded, both models analyze it. The system compares their confidence scores and selects the prediction with the highest confidence.
-   **Context-Aware Reporting:**
    -   The LLM prompt generation was updated to be dynamic.
    -   If the winner is an Alzheimer's class (e.g., "MildDemented"), the prompt instructs the LLM to act as a Neuroradiologist looking for atrophy and ventricular enlargement.
    -   If the winner is a Tumor class, it retains the original tumor-focused prompt.
-   **Bug Fixes:**
    -   Addressed `torch.load` security warnings by explicitly setting `weights_only=False` for the new model.
    -   Fixed syntax errors in the multi-line f-strings for the LLM prompts.

### Session Summary: Comprehensive Data Visualization Suite

This session added a robust performance analysis and visualization suite to validate the system's "Competitive Classification" logic.

**Key Deliverables:**
-   **New Directory:** `data_visualization/` containing the analysis tools.
-   **Performance Script:** `visualize_performance.py`:
    -   Loads both the Brain Tumor and Alzheimer's models.
    -   Runs evaluation on **20%** of the dataset (randomly subsampled for efficiency).
    -   Simulates the dual-model decision logic used in the GUI to verify cross-domain accuracy.
    -   Generates performance metrics and saves them as images.
- **Visualizations Created:**
    -   **Confusion Matrices:** Heatmaps showing classification accuracy (in percentages) for the Tumor model, Alzheimer's model, and the Combined System.
    -   **Accuracy Bar Charts:** Per-class accuracy breakdowns for all scenarios.
-   **Results:** The combined system demonstrated ~76% overall accuracy and ~79% accuracy in correctly selecting the appropriate model type (Tumor vs. Alzheimer's) for a given image.

### Session Summary: Hierarchical Classification & Gatekeeper Integration

This session introduced a "Gatekeeper" model to transition from a competitive classification approach to a more robust hierarchical one.

**Key Achievements:**
-   **Gatekeeper Model:**
    -   Implemented `src/gatekeeper_model.py` using a **ResNet50** backbone.
    -   Designed as a binary classifier to distinguish between "Tumor" and "Dementia" MRI scans.
    -   Trained via `src/train_gatekeeper.py` to act as the primary router for the system.
-   **Hierarchical Architecture:**
    -   Updated the analysis pipeline to first run the Gatekeeper model.
    -   The Gatekeeper's output now determines which specialized model (EfficientNet-B3 for tumors or MobileNetV3 for Alzheimer's) should perform the final diagnosis.
    -   This reduces false positives by ensuring the specialized models only process images relevant to their training domain.
-   **GUI Integration:**
    -   Updated `src/radiology_report_gui.py` to load and utilize the Gatekeeper model during the "Analyze" phase.
    -   The GUI now reports the Gatekeeper's routing decision in the logs/console.
-   **Improved Reliability:** The hierarchical approach addresses edge cases where one specialized model might incorrectly assign high confidence to an image from the "other" domain.

