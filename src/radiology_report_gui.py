import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, mobilenet_v3_large
from torchvision import transforms
from src.gatekeeper_model import GatekeeperClassifier
from src.grounded_report import build_grounded_report, render_report_html, render_report_markdown
from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image
try:
    from src.experiment_pipeline import build_model as build_pipeline_model
    from src.experiment_pipeline import build_transforms as build_pipeline_transforms
except Exception:
    build_pipeline_model = None
    build_pipeline_transforms = None
import ollama
import threading
import queue
import io
import base64
from datetime import datetime
from PIL import Image, ImageTk
from tkhtmlview import HTMLLabel
from tkcalendar import DateEntry
from weasyprint import HTML as WeasyHTML
from pypdf import PdfReader, PdfWriter

# WORKAROUND: Map 'gatekeeper_model' to 'src.gatekeeper_model' so torch.load finds it
import src.gatekeeper_model
sys.modules['gatekeeper_model'] = src.gatekeeper_model

APP_TITLE = "Radiology Report Generator"
BINARY_ROUTER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "binary_router.pt")
GATEKEEPER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gatekeeper_classifier.pt")
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "brain_tumor_classifier.pt")
ALZHEIMERS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alzheimers_classifier.pt")

# Constants for Alzheimer's
ALZ_CLASSES_3 = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']
ALZ_CLASSES_4 = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
ALZ_CLASSES = ALZ_CLASSES_4
TUMOR_CLASSES_3 = ["glioma", "meningioma", "pituitary"]
TUMOR_CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]
ALZ_IMG_SIZE = 224

def build_tumor_model(arch: str, num_classes: int):
    if arch == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def build_alzheimers_model(arch: str, num_classes: int):
    if arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def infer_linear_outputs(model, fallback):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features
    return fallback

def get_tumor_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_alzheimers_transform():
    # Matches the MobileNetV3 training (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_gatekeeper_transform():
    # Gatekeeper (ResNet50) uses 224x224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    model_name: str
    source_image: Image.Image
    cam_model: Optional[nn.Module] = None
    cam_transform: Optional[Callable] = None
    cam_class_idx: Optional[int] = None


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill="both", expand=True)
        self.master = master

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            try:
                import torch_directml as dml
                self.device = dml.device()
            except Exception:
                self.device = torch.device("cpu")

        # Models
        self.gatekeeper_model = None
        self.gatekeeper_mode = "legacy_3way"
        self.gatekeeper_classes = ["Normal", "tumor", "dementia"]
        self.tumor_model = None
        self.tumor_classes = []
        self.alz_model = None
        self.alz_classes = ALZ_CLASSES_4

        self.image_path = None
        self.tk_image = None
        self._original_pil_image = None
        self.heatmap_pil_image = None
        self.showing_heatmap = False
        self.last_model_name = None

        # Transforms
        self.tumor_tfms = get_tumor_transform()
        self.alz_tfms = get_alzheimers_transform()
        self.gate_tfms = get_gatekeeper_transform()

        # Reporting
        self.report_queue = queue.Queue()
        self.last_report_html = None
        self.last_report_markdown = None

        self._build_ui()
        
        # Auto-load models
        self.load_models()

    def _build_ui(self):
        # Top bar for model status
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", pady=(0, 10))
        self.model_status_label = ttk.Label(top_bar, text="Models: Loading...", anchor="w")
        self.model_status_label.pack(side="left", fill="x", expand=True)
        
        # Manual reload button
        ttk.Button(top_bar, text="Reload Models", command=self.load_models).pack(side="right")

        # Main layout with a resizable pane
        main_pane = ttk.PanedWindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        # --- Left Column: Image and Classification ---
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        # 1. Input Notebook (Top)
        input_notebook = ttk.Notebook(left_frame)
        input_notebook.pack(side="top", fill="x", pady=(0, 10))

        # Tab 1: Patient Info
        patient_frame = ttk.Frame(input_notebook, padding=10)
        input_notebook.add(patient_frame, text="Patient Info")
        
        info_grid = ttk.Frame(patient_frame)
        info_grid.pack(fill="x", expand=True)
        info_grid.columnconfigure(1, weight=1)

        ttk.Label(info_grid, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.patient_name_entry = ttk.Entry(info_grid)
        self.patient_name_entry.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(info_grid, text="Patient ID:").grid(row=1, column=0, sticky="w", pady=2)
        self.patient_id_entry = ttk.Entry(info_grid)
        self.patient_id_entry.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(info_grid, text="DOB:").grid(row=2, column=0, sticky="w", pady=2)
        self.dob_entry = DateEntry(info_grid, date_pattern='yyyy-mm-dd', width=12, background='darkblue', foreground='white', borderwidth=2,
                                   selectmode='day', year=datetime.now().year - 40, month=1, day=1)
        self.dob_entry.grid(row=2, column=1, sticky="w", padx=5)

        # Tab 2: Exam Details
        exam_frame = ttk.Frame(input_notebook, padding=10)
        input_notebook.add(exam_frame, text="Exam Details")
        
        # Control Buttons for Exam Details
        self.exam_btn_frame = ttk.Frame(exam_frame)
        self.exam_btn_frame.pack(fill="x", pady=(0, 10))
        
        self.manual_btn = ttk.Button(self.exam_btn_frame, text="Manual Entry", command=self.enable_manual_input, state="disabled")
        self.manual_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.ai_fill_btn = ttk.Button(self.exam_btn_frame, text="AI Auto-Detect", command=self.auto_detect_exam_details, state="disabled")
        self.ai_fill_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        exam_grid = ttk.Frame(exam_frame)
        exam_grid.pack(fill="x", expand=True)
        exam_grid.columnconfigure(1, weight=1)

        ttk.Label(exam_grid, text="Reason:").grid(row=0, column=0, sticky="w", pady=2)
        self.reason_entry = ttk.Entry(exam_grid, state="disabled")
        self.reason_entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="History:").grid(row=1, column=0, sticky="w", pady=2)
        self.history_entry = ttk.Entry(exam_grid, state="disabled")
        self.history_entry.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(exam_grid, text="Comparison:").grid(row=2, column=0, sticky="w", pady=2)
        self.comparison_entry = ttk.Entry(exam_grid, state="disabled")
        self.comparison_entry.grid(row=2, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="Technique:").grid(row=3, column=0, sticky="w", pady=2)
        self.technique_entry = ttk.Entry(exam_grid, state="disabled")
        self.technique_entry.grid(row=3, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="Contrast:").grid(row=4, column=0, sticky="w", pady=2)
        self.contrast_entry = ttk.Entry(exam_grid, state="disabled")
        self.contrast_entry.grid(row=4, column=1, sticky="ew", padx=5)

        # 5. Classification Result (Bottom)
        result_frame = ttk.LabelFrame(left_frame, text="Classification Details", padding=10)
        result_frame.pack(side="bottom", fill="x")
        self.pred_label = ttk.Label(result_frame, text="Prediction: -", font=("Segoe UI", 11, "bold"))
        self.pred_label.pack(anchor="w")
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -")
        self.confidence_label.pack(anchor="w")
        self.model_used_label = ttk.Label(result_frame, text="Model Used: -", foreground="#555")
        self.model_used_label.pack(anchor="w")

        # 4. Progress Bar (Bottom, above results)
        self.progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress_bar.pack(side="bottom", fill="x", pady=5)
        self.progress_bar.pack_forget() # Hidden initially

        # 3. Analysis Controls (Bottom, above progress bar)
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding=10)
        analysis_frame.pack(side="bottom", fill="x", pady=10)
        
        self.generate_btn = ttk.Button(analysis_frame, text="Analyze & Generate Report", command=self.on_analyze_and_generate, state="disabled")
        self.generate_btn.pack(fill="x", expand=True)

        # 2. Image/Scan Frame (Middle - Expands)
        img_frame = ttk.LabelFrame(left_frame, text="Scan", padding=10)
        img_frame.pack(side="top", fill="both", expand=True)
        
        # Scan Button (Upload) - Pack to BOTTOM first so it stays visible
        self.choose_img_btn = ttk.Button(img_frame, text="Scan", command=self.on_choose_image)
        self.choose_img_btn.pack(side="bottom", fill="x", pady=(5, 0))

        # Grad-CAM toggle + caption (disabled until a heatmap is generated)
        self.heatmap_caption_label = ttk.Label(
            img_frame, text="", foreground="#555", font=("Segoe UI", 8), wraplength=320, justify="left"
        )
        self.heatmap_caption_label.pack(side="bottom", fill="x", pady=(2, 0))

        self.heatmap_toggle_btn = ttk.Button(
            img_frame, text="Show Heatmap", command=self._toggle_heatmap_view, state="disabled"
        )
        self.heatmap_toggle_btn.pack(side="bottom", fill="x", pady=(5, 0))

        # Canvas - Pack to TOP and expand to fill remaining space
        self.canvas = tk.Canvas(img_frame, width=350, height=350, bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.canvas.pack(side="top", fill="both", expand=True)

        # --- Right Column: Generated Report ---
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=2)
        
        report_frame = ttk.LabelFrame(right_frame, text="Generated Radiology Report", padding=10)
        report_frame.pack(fill="both", expand=True)
        
        self.report_html = HTMLLabel(report_frame, html="<p>Report will be generated here.</p>")
        self.report_html.pack(fill="both", expand=True)
        
        self.save_pdf_btn = ttk.Button(right_frame, text="Save as PDF", command=self.on_save_pdf, state="disabled")
        self.save_pdf_btn.pack(pady=5)
    def load_models(self):
        status_texts = []
        
        # 1. Load Binary Router if present, otherwise fall back to legacy 3-way gatekeeper.
        self.gatekeeper_model = None
        self.gatekeeper_mode = "legacy_3way"
        self.gatekeeper_classes = ["Normal", "tumor", "dementia"]

        if os.path.isfile(BINARY_ROUTER_MODEL_PATH) and build_pipeline_model:
            try:
                checkpoint = torch.load(BINARY_ROUTER_MODEL_PATH, map_location=self.device, weights_only=False)
                if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                    raise ValueError("binary_router.pt is missing pipeline checkpoint metadata")

                self.gatekeeper_classes = checkpoint.get("class_names", ["tumor", "dementia"])
                model = build_pipeline_model(
                    checkpoint.get("arch", "resnet50"),
                    len(self.gatekeeper_classes),
                    pretrained=False,
                )
                model.load_state_dict(checkpoint["model_state"])
                model.eval().to(self.device)
                self.gatekeeper_model = model
                self.gatekeeper_mode = "binary"

                if build_pipeline_transforms:
                    self.gate_tfms = build_pipeline_transforms(
                        train=False,
                        image_size=int(checkpoint.get("image_size", 224)),
                    )
                status_texts.append("Binary Router: Ready")
            except Exception as e:
                print(f"Error loading binary router model: {e}")
                status_texts.append("Binary Router: Error")

        elif os.path.isfile(GATEKEEPER_MODEL_PATH):
            try:
                # Load the file first
                loaded_obj = torch.load(GATEKEEPER_MODEL_PATH, map_location=self.device, weights_only=False)
                
                if isinstance(loaded_obj, nn.Module):
                    # It's a full model
                    self.gatekeeper_model = loaded_obj
                else:
                    # It's a state dict
                    model = GatekeeperClassifier(num_classes=3, freeze_base=False)
                    model.load_state_dict(loaded_obj)
                    self.gatekeeper_model = model
                
                self.gatekeeper_model.eval().to(self.device)
                status_texts.append("Gatekeeper: Ready")
            except Exception as e:
                print(f"Error loading gatekeeper model: {e}")
                status_texts.append("Gatekeeper: Error")
        else:
            status_texts.append("Binary Router/Gatekeeper: Not Found")

        # 2. Load Tumor Model
        if os.path.isfile(TUMOR_MODEL_PATH):
            try:
                checkpoint = torch.load(TUMOR_MODEL_PATH, map_location=self.device, weights_only=False)
                
                # Check if it's a dictionary (checkpoint) or a full model
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    arch = checkpoint.get("arch", "efficientnet_b3")
                    self.tumor_classes = checkpoint.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
                    model = build_tumor_model(arch, len(self.tumor_classes))
                    model.load_state_dict(checkpoint["model_state"])
                else:
                    # Assume it's a full model object or state dict
                    # If it's a state dict, we need to know the arch. If it's a full model, just use it.
                    if isinstance(checkpoint, nn.Module):
                        model = checkpoint
                        output_count = infer_linear_outputs(model, 4)
                        self.tumor_classes = TUMOR_CLASSES_3 if output_count == 3 else TUMOR_CLASSES_4
                    else:
                        # Fallback for state dict only
                        model = build_tumor_model("efficientnet_b3", 4)
                        model.load_state_dict(checkpoint)
                        self.tumor_classes = TUMOR_CLASSES_4

                model.eval().to(self.device)
                self.tumor_model = model
                status_texts.append("Tumor: Ready")
            except Exception as e:
                print(f"Error loading tumor model: {e}")
                status_texts.append("Tumor: Error")
        else:
            status_texts.append("Tumor: Not Found")

        # 3. Load Alzheimer's Model
        if os.path.isfile(ALZHEIMERS_MODEL_PATH):
            try:
                checkpoint = torch.load(ALZHEIMERS_MODEL_PATH, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    arch = checkpoint.get("arch", "mobilenet_v3_large")
                    self.alz_classes = checkpoint.get("class_names", ALZ_CLASSES_4)
                    self.alz_model = build_alzheimers_model(arch, len(self.alz_classes))
                    self.alz_model.load_state_dict(checkpoint["model_state"])
                else:
                    # Older training scripts saved the full model object.
                    self.alz_model = checkpoint
                    output_count = infer_linear_outputs(self.alz_model, 4)
                    self.alz_classes = ALZ_CLASSES_3 if output_count == 3 else ALZ_CLASSES_4
                self.alz_model.eval().to(self.device)
                status_texts.append("Alzheimer's: Ready")
            except Exception as e:
                print(f"Error loading alzheimer model: {e}")
                status_texts.append("Alzheimer's: Error")
        else:
            status_texts.append("Alzheimer's: Not Found")

        self.model_status_label.configure(text=" | ".join(status_texts))
        
        if self.tumor_model or self.alz_model:
            if self.image_path:
                self.generate_btn.configure(state="normal")

    def on_choose_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[["Image files", "*.jpg;*.jpeg;*.png;*.bmp"]],
            initialdir=os.path.abspath(os.path.join("data", "evaluation", "images")),
        )
        if path:
            self.image_path = path
            self.heatmap_pil_image = None
            self.showing_heatmap = False
            self.heatmap_toggle_btn.configure(text="Show Heatmap", state="disabled")
            self.heatmap_caption_label.configure(text="")
            self._display_image(path)

            # Enable Exam Detail Controls
            self.manual_btn.configure(state="normal")
            self.ai_fill_btn.configure(state="normal")

    def enable_manual_input(self):
        # Unlock fields and fill with generic defaults if empty
        widgets = [self.reason_entry, self.history_entry, self.comparison_entry, self.technique_entry, self.contrast_entry]
        defaults = ["Neurological symptoms", "Not provided", "None", "Standard MRI Brain Protocol", "None"]
        
        for w, d in zip(widgets, defaults):
            w.configure(state="normal")
            if not w.get():
                w.insert(0, d)
        
        # Now allow analysis
        self.generate_btn.configure(state="normal")

    def auto_detect_exam_details(self):
        if not self.image_path: return
        
        self.ai_fill_btn.configure(text="Detecting...", state="disabled")
        self.manual_btn.configure(state="disabled")
        
        # Run in thread
        thread = threading.Thread(target=self._run_ai_autodetect, daemon=True)
        thread.start()

    def _run_ai_autodetect(self):
        try:
            image_bytes = None
            with open(self.image_path, "rb") as f:
                image_bytes = f.read()

            prompt = (
                "Analyze only basic MRI acquisition metadata from the image appearance. "
                "If a detail is not directly supported, return 'Not provided'. "
                "Do not infer diagnosis, pathology, lesion size, lesion location, or a clinical reason from visible abnormalities. "
                "Return valid JSON only:\n"
                "{\n"
                '  "technique": "string",\n'
                '  "contrast": "string",\n'
                '  "reason": "Not provided",\n'
                '  "comparison": "None"\n'
                "}"
            )

            response = ollama.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }],
            )
            
            # Parse JSON
            import json
            import re
            content = response['message']['content']
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            json_str = match.group(1) if match else content
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str) # Sanitize
            
            data = json.loads(json_str)
            
            # Update UI in main thread
            self.master.after(0, lambda: self._update_exam_fields(data))

        except Exception as e:
            print(f"Auto-detect error: {e}")
            # Fallback to manual
            self.master.after(0, lambda: self._update_exam_fields(None))

    def _update_exam_fields(self, data):
        self.ai_fill_btn.configure(text="AI Auto-Detect", state="normal")
        self.manual_btn.configure(state="normal")
        self.enable_manual_input() # Unlock first and enable generate_btn
        
        if data:
            self.technique_entry.delete(0, tk.END)
            self.technique_entry.insert(0, data.get("technique", "Standard MRI"))
            
            self.contrast_entry.delete(0, tk.END)
            self.contrast_entry.insert(0, data.get("contrast", "None"))
            
            self.reason_entry.delete(0, tk.END)
            self.reason_entry.insert(0, data.get("reason", "Evaluation"))
            
            self.comparison_entry.delete(0, tk.END)
            self.comparison_entry.insert(0, data.get("comparison", "None"))
        else:
            messagebox.showwarning("AI Detection", "Could not auto-detect details. Switched to manual mode.")
        
        # Ensure analysis is enabled
        self.generate_btn.configure(state="normal")

    def _display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            self._original_pil_image = img
            self._render_pil_image(img)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

    def _render_pil_image(self, img: Image.Image):
        # Force layout update to get correct dimensions
        self.master.update_idletasks()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1 or h <= 1: w, h = 350, 350
        thumb = img.copy()
        thumb.thumbnail((w, h))
        self.tk_image = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.tk_image)

    def _toggle_heatmap_view(self):
        if self.heatmap_pil_image is None or self._original_pil_image is None:
            return
        self.showing_heatmap = not self.showing_heatmap
        if self.showing_heatmap:
            self._render_pil_image(self.heatmap_pil_image)
            self.heatmap_toggle_btn.configure(text="Show Original")
        else:
            self._render_pil_image(self._original_pil_image)
            self.heatmap_toggle_btn.configure(text="Show Heatmap")

    def on_analyze_and_generate(self):
        if not self.image_path: return

        self.generate_btn.configure(state="disabled")
        self.choose_img_btn.configure(state="disabled")
        self.save_pdf_btn.configure(state="disabled")
        self.pred_label.configure(text="Prediction: Analyzing...")
        self.confidence_label.configure(text="Confidence: -")
        self.model_used_label.configure(text="Model Used: -")
        self.report_html.set_html("<p><i>Running multi-stage analysis and generating grounded report...</i></p>")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start()
        self.update_idletasks()

        try:
            result = self._run_cascaded_classification()
            pred_label, pred_conf, model_name = result.label, result.confidence, result.model_name
            self.last_model_name = model_name

            self.pred_label.configure(text=f"Prediction: {pred_label}")
            self.confidence_label.configure(text=f"Confidence: {pred_conf:.2f}%")
            self.model_used_label.configure(text=f"Model Used: {model_name}")

            self.heatmap_pil_image = self._run_grad_cam(result)
            self.showing_heatmap = False
            if self.heatmap_pil_image is not None:
                self.showing_heatmap = True
                self._render_pil_image(self.heatmap_pil_image)
                self.heatmap_toggle_btn.configure(text="Show Original", state="normal")
                self.heatmap_caption_label.configure(
                    text=(
                        f"Grad-CAM: regions that most influenced the {model_name} prediction. "
                        "Not a diagnostic localization of pathology."
                    )
                )
            else:
                self.heatmap_toggle_btn.configure(text="Show Heatmap", state="disabled")
                self.heatmap_caption_label.configure(text="")

            confidence_str = f"{pred_conf:.1f}%"
            patient_info = {
                "name": self.patient_name_entry.get(),
                "dob": self.dob_entry.get_date().strftime('%Y-%m-%d'),
                "patient_id": self.patient_id_entry.get(),
            }

            # Collect user inputs
            user_inputs = {
                "reason": self.reason_entry.get(),
                "history": self.history_entry.get(),
                "comparison": self.comparison_entry.get(),
                "technique": self.technique_entry.get(),
                "contrast": self.contrast_entry.get()
            }

            self._start_report_generation(pred_label, confidence_str, patient_info, user_inputs, model_name)

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred:\n{e}")
            self.generate_btn.configure(state="normal")
            self.choose_img_btn.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

    @torch.no_grad()
    def _run_cascaded_classification(self):
        img = Image.open(self.image_path).convert("RGB")
        
        # 1. Gatekeeper Phase (Routing)
        target_domain = "tumor" # Default fallback
        gate_conf = 0.0

        if self.gatekeeper_model:
            # ResNet50 typically uses 224x224
            tensor = self.gate_tfms(img).unsqueeze(0).to(self.device)
            # Use appropriate autocast for the device
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    output = self.gatekeeper_model(tensor)
            else:
                output = self.gatekeeper_model(tensor)

            if self.gatekeeper_mode == "binary":
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = torch.topk(probs, 1)
                class_idx = top_idx.item()
                gate_conf = top_prob.item()
                target_domain = self.gatekeeper_classes[class_idx].lower()
                print(f"Binary router: {target_domain} scan detected (confidence {gate_conf*100:.2f}%)")
            else:
                # Legacy multi-class output: [Normal, Tumor, Dementia]
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = torch.topk(probs, 1)
                class_idx = top_idx.item()
                gate_conf = top_prob.item()

                if class_idx == 0:
                    target_domain = "normal"
                    print(f"Gatekeeper: Normal/No Tumor detected (confidence {gate_conf*100:.2f}%)")
                    return ClassificationResult("Normal", gate_conf * 100.0, "Gatekeeper Model", img)
                elif class_idx == 1:
                    target_domain = "tumor"
                    print(f"Gatekeeper: Tumor scan detected (confidence {gate_conf*100:.2f}%)")
                elif class_idx == 2:
                    target_domain = "dementia"
                    print(f"Gatekeeper: Dementia detected (confidence {gate_conf*100:.2f}%)")

        # 2. Specialized Classification Phase
        if target_domain == "dementia" and self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.alz_model(tensor_alz)
            else:
                logits = self.alz_model(tensor_alz)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's/Dementia Model", img,
                cam_model=self.alz_model, cam_transform=self.alz_tfms, cam_class_idx=top_idx.item(),
            )

        elif target_domain == "tumor" and self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.tumor_model(tensor)
            else:
                logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model", img,
                cam_model=self.tumor_model, cam_transform=self.tumor_tfms, cam_class_idx=top_idx.item(),
            )

        # Fallback if specific model fails but we have a general prediction or fallback model
        if target_domain == "normal":
             return ClassificationResult("Normal", gate_conf * 100.0, "Gatekeeper Model", img)

        # Fallback if preferred model isn't loaded but the other is
        if self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model (Fallback)", img,
                cam_model=self.tumor_model, cam_transform=self.tumor_tfms, cam_class_idx=top_idx.item(),
            )

        if self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            logits = self.alz_model(tensor_alz)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's Model (Fallback)", img,
                cam_model=self.alz_model, cam_transform=self.alz_tfms, cam_class_idx=top_idx.item(),
            )

        raise Exception("No models loaded or available.")

    def _run_grad_cam(self, result: "ClassificationResult") -> Optional[Image.Image]:
        if result.cam_model is None or result.cam_transform is None or result.cam_class_idx is None:
            return None
        try:
            target_layer = get_target_layer(result.cam_model)
            tensor = result.cam_transform(result.source_image).unsqueeze(0).to(self.device)
            cam = GradCAM(result.cam_model, target_layer).generate(tensor, result.cam_class_idx)
            if cam is None:
                return None
            return overlay_cam_on_image(result.source_image, cam)
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return None

    def _start_report_generation(self, prediction, confidence_str, patient_info, user_inputs, model_name):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(prediction, confidence_str, patient_info, user_inputs, model_name),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, prediction, confidence_str, patient_info, user_inputs, model_name):
        try:
            report_date = datetime.now().strftime("%B %d, %Y")
            report = build_grounded_report(
                prediction=prediction,
                confidence=confidence_str,
                patient_info=patient_info,
                exam_details=user_inputs,
                report_date=report_date,
                model_used=model_name,
                source_image=self.image_path,
            )
            html_text = render_report_html(report)
            self.last_report_html = html_text
            self.last_report_markdown = render_report_markdown(report)
            self.report_queue.put(html_text)

        except Exception as e:
            self.report_queue.put(f"<p><b>Error generating report:</b></p><pre>{e}</pre>")

    def _check_report_queue(self):
        try:
            html_result = self.report_queue.get_nowait()
            self.last_report_html = html_result
            self.report_html.set_html(html_result)
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.generate_btn.configure(state="normal")
            self.choose_img_btn.configure(state="normal")
            self.save_pdf_btn.configure(state="normal")
        except queue.Empty:
            self.master.after(100, self._check_report_queue)

    def on_save_pdf(self):
        if not self.last_report_html: return

        filepath = filedialog.asksaveasfilename(
            title="Save Report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not filepath: return

        password = simpledialog.askstring("PDF Encryption", "Enter a password (optional):", show='*')

        try:
            # Use the already formatted HTML
            pdf_html = self.last_report_html
            
            # Embed image
            image_data_uri = None
            try:
                with open(self.image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    image_data_uri = f"data:image/jpeg;base64,{encoded}"
            except Exception: pass

            if image_data_uri:
                pdf_html += f'<div style="text-align: center; margin-top: 20px;"><img src="{image_data_uri}" alt="MRI Scan" style="max-width: 300px; height: auto;"></div>'

            # Embed Grad-CAM heatmap, if one was generated for this analysis
            if self.heatmap_pil_image is not None:
                try:
                    heatmap_buffer = io.BytesIO()
                    self.heatmap_pil_image.save(heatmap_buffer, format="PNG")
                    heatmap_data_uri = "data:image/png;base64," + base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
                    model_label = self.last_model_name or "specialist"
                    pdf_html += (
                        '<div style="text-align: center; margin-top: 20px;">'
                        f'<img src="{heatmap_data_uri}" alt="Grad-CAM heatmap" style="max-width: 300px; height: auto;">'
                        '<p style="font-size: 0.85em; color: #555; max-width: 400px; margin: 6px auto 0;">'
                        f'Grad-CAM visualization: highlights the image regions that most influenced the {model_label} '
                        'classifier\'s predicted class. This is a visualization of model attention, not a validated '
                        'diagnostic localization of pathology, and must be interpreted only alongside direct '
                        'radiologist review of the original MRI images.'
                        '</p></div>'
                    )
                except Exception:
                    pass

            # Add CSS for Page Numbers in WeasyPrint
            css = """
            @page {
                size: Letter;
                margin: 2cm;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                    font-family: sans-serif;
                    font-size: 9pt;
                    color: #555;
                }
            }
            body { font-family: sans-serif; font-size: 10pt; } 
            h1, h2, h3 { color: #333; } 
            table { border-collapse: collapse; width: 100%; } 
            td, th { padding: 4px; text-align: left; border-bottom: 1px solid #ddd; }
            """
            html_with_style = f"<style>{css}</style>{pdf_html}"
            
            pdf_buffer = io.BytesIO()
            WeasyHTML(string=html_with_style, base_url=self.image_path).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)

            if password:
                reader = PdfReader(pdf_buffer)
                writer = PdfWriter()
                for page in reader.pages: writer.add_page(page)
                writer.encrypt(password)
                with open(filepath, "wb") as f: writer.write(f)
            else:
                with open(filepath, "wb") as f: f.write(pdf_buffer.read())
            
            messagebox.showinfo("Success", f"Report saved to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("PDF Error", f"Failed to save PDF:\n{e}")

def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1100x750")
    try:
        style = ttk.Style()
        style.theme_use("vista" if sys.platform.startswith("win") else "clam")
    except:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
