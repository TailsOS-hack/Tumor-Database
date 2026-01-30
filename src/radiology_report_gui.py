import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, mobilenet_v3_large
from torchvision import transforms
from src.gatekeeper_model import GatekeeperClassifier
import ollama
import threading
import queue
import io
import base64
from datetime import datetime
from PIL import Image, ImageTk
from tkhtmlview import HTMLLabel
from tkcalendar import DateEntry
import markdown
from weasyprint import HTML as WeasyHTML
from pypdf import PdfReader, PdfWriter

# WORKAROUND: Map 'gatekeeper_model' to 'src.gatekeeper_model' so torch.load finds it
import src.gatekeeper_model
sys.modules['gatekeeper_model'] = src.gatekeeper_model

APP_TITLE = "Radiology Report Generator"
GATEKEEPER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gatekeeper_classifier.pt")
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "brain_tumor_classifier.pt")
ALZHEIMERS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alzheimers_classifier.pt")

# Constants for Alzheimer's
ALZ_CLASSES = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']
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
        self.tumor_model = None
        self.tumor_classes = []
        self.alz_model = None
        self.alz_classes = ALZ_CLASSES

        self.image_path = None
        self.tk_image = None
        
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
        
        # 1. Load Gatekeeper Model
        if os.path.isfile(GATEKEEPER_MODEL_PATH):
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
            status_texts.append("Gatekeeper: Not Found")

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
                        # Try to infer classes or use default
                        self.tumor_classes = ["glioma", "meningioma", "notumor", "pituitary"]
                    else:
                        # Fallback for state dict only
                        model = build_tumor_model("efficientnet_b3", 4)
                        model.load_state_dict(checkpoint)
                        self.tumor_classes = ["glioma", "meningioma", "notumor", "pituitary"]

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
                # This model was saved as a full object, not a state dict
                self.alz_model = torch.load(ALZHEIMERS_MODEL_PATH, map_location=self.device, weights_only=False)
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
                "Analyze this MRI scan metadata from the image appearance. "
                "Guess the likely imaging technique (e.g. 'Axial T2-weighted MRI', 'Sagittal FLAIR'), "
                "whether contrast was used, and a plausible reason for exam based on visible abnormalities. "
                "Return valid JSON only:\n"
                "{\n"
                '  "technique": "string",\n'
                '  "contrast": "string",\n'
                '  "reason": "string",\n'
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
            # Force layout update to get correct dimensions
            self.master.update_idletasks()
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w <= 1 or h <= 1: w, h = 350, 350
            img.thumbnail((w, h))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self.tk_image)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

    def on_analyze_and_generate(self):
        if not self.image_path: return

        self.generate_btn.configure(state="disabled")
        self.choose_img_btn.configure(state="disabled")
        self.save_pdf_btn.configure(state="disabled")
        self.pred_label.configure(text="Prediction: Analyzing...")
        self.confidence_label.configure(text="Confidence: -")
        self.model_used_label.configure(text="Model Used: -")
        self.report_html.set_html("<p><i>Running multi-stage analysis and generating report...</i></p>")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start()
        self.update_idletasks()

        try:
            pred_label, pred_conf, model_name = self._run_cascaded_classification()
            
            self.pred_label.configure(text=f"Prediction: {pred_label}")
            self.confidence_label.configure(text=f"Confidence: {pred_conf:.2f}%")
            self.model_used_label.configure(text=f"Model Used: {model_name}")
            
            confidence_str = f"{pred_conf:.1f}%"
            patient_info_formatted = (
                f"**Name:** {self.patient_name_entry.get()}<br>"
                f"**DOB:** {self.dob_entry.get_date().strftime('%Y-%m-%d')}<br>"
                f"**Patient ID:** {self.patient_id_entry.get()}"
            )

            # Collect user inputs
            user_inputs = {
                "reason": self.reason_entry.get(),
                "history": self.history_entry.get(),
                "comparison": self.comparison_entry.get(),
                "technique": self.technique_entry.get(),
                "contrast": self.contrast_entry.get()
            }

            self._start_report_generation(pred_label, confidence_str, patient_info_formatted, user_inputs)

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
            
            # Multi-class output: [Normal, Tumor, Dementia]
            probs = torch.softmax(output, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            class_idx = top_idx.item()
            gate_conf = top_prob.item()

            if class_idx == 0:
                target_domain = "normal"
                print(f"Gatekeeper: Normal/No Tumor detected (confidence {gate_conf*100:.2f}%)")
                return "Normal", gate_conf * 100.0, "Gatekeeper Model"
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
            return self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's/Dementia Model"
            
        elif target_domain == "tumor" and self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.tumor_model(tensor)
            else:
                logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model"
        
        # Fallback if specific model fails but we have a general prediction or fallback model
        if target_domain == "normal":
             return "Normal", gate_conf * 100.0, "Gatekeeper Model"

        # Fallback if preferred model isn't loaded but the other is
        if self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model (Fallback)"
        
        if self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            logits = self.alz_model(tensor_alz)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's Model (Fallback)"

        raise Exception("No models loaded or available.")

    def _start_report_generation(self, prediction, confidence_str, patient_info_formatted, user_inputs):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(prediction, confidence_str, patient_info_formatted, user_inputs),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, prediction, confidence_str, patient_info_formatted, user_inputs):
        try:
            report_date = datetime.now().strftime("%B %d, %Y")
            
            image_bytes = None
            try:
                with open(self.image_path, "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                print(f"Image read error: {e}")

            prompt = self._create_llava_prompt(prediction, confidence_str, patient_info_formatted, report_date)
            
            response = ollama.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes] if image_bytes else []
                }],
            )
            
            raw_content = response['message']['content']
            
            # extract JSON
            import json
            import re
            
            json_str = raw_content
            # Try to find JSON block if wrapped in markdown
            match = re.search(r'```json\s*(.*?)\s*```', raw_content, re.DOTALL)
            if match:
                json_str = match.group(1)
            
            # Sanitization: Remove trailing commas
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            try:
                data = json.loads(json_str)
                html_text = self._format_json_to_html(data, patient_info_formatted, report_date, prediction, confidence_str, user_inputs)
                self.last_report_html = html_text # Save for PDF
                self.report_queue.put(html_text)
            except json.JSONDecodeError:
                print("JSON Decode Failed. Raw output:", raw_content)
                self.last_report_markdown = raw_content
                html_text = markdown.markdown(raw_content, extensions=['fenced_code', 'tables'])
                self.report_queue.put(f"<div style='color:red'><b>Warning: AI did not return strict JSON. Raw output shown:</b></div><br>{html_text}")

        except Exception as e:
            self.report_queue.put(f"<p><b>Error generating report:</b></p><pre>{e}</pre>")

    def _format_json_to_html(self, data, patient_info, date, pred, conf, user_inputs):
        # Helper to safely get fields
        def g(path, default="-"):
            val = data
            for key in path:
                if isinstance(val, dict):
                    val = val.get(key, default)
                else:
                    return default
            return val

        # Override JSON values with User Inputs
        reason = user_inputs.get("reason") or g(['clinical_information', 'reason_for_exam'])
        history = user_inputs.get("history") or g(['clinical_information', 'clinical_history'])
        comparison = user_inputs.get("comparison") or g(['clinical_information', 'comparison'])
        technique = user_inputs.get("technique") or g(['procedure_details', 'technique'])
        contrast = user_inputs.get("contrast") or g(['procedure_details', 'contrast_info', 'agent'])

        # Timestamp for footer
        print_ts = datetime.now().strftime("%m/%d/%Y %H:%M EST")

        # Construct the HTML Report
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0;">Final Report</h1>
                <p style="margin: 5px 0;"><b>* Final Report *</b></p>
            </div>

            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <td style="vertical-align: top; width: 50%; padding-right: 10px;">
                        <h3>Patient Information</h3>
                        {patient_info.replace('**', '<b>').replace('**', '</b>')}
                    </td>
                    <td style="vertical-align: top; width: 50%;">
                        <h3>Report Details</h3>
                        <b>Date:</b> {date}<br>
                        <b>AI Analysis:</b> {pred} ({conf})
                    </td>
                </tr>
            </table>

            <div style="margin-bottom: 15px;">
                <b>Reason for Exam:</b> {reason}<br>
                <b>Clinical History:</b> {history}<br>
                <b>Comparison:</b> {comparison}
            </div>

            <div style="margin-bottom: 15px;">
                <b>Technique:</b> {technique}<br>
                <b>Contrast:</b> {contrast}
            </div>

            <hr style="border: 0; border-top: 1px solid #ccc; margin: 20px 0;">

            <h3>FINDINGS</h3>
            <p>
            <b>Cerebral Parenchyma:</b> {g(['findings', 'cerebral_parenchyma'])}<br><br>
            <b>Extra-axial Spaces:</b> {g(['findings', 'extra_axial_spaces'])}<br><br>
            <b>Ventricles:</b> {g(['findings', 'ventricles'])}<br><br>
            <b>Mass Effect:</b> {g(['findings', 'mass_effect'])}<br><br>
            <b>Vascular Structures:</b> {g(['findings', 'vascular_structures'])}<br><br>
            <b>Bones & Soft Tissues:</b> {g(['findings', 'bones_and_soft_tissues'])}<br><br>
            <b>Paranasal Sinuses/Mastoids:</b> {g(['findings', 'paranasal_sinuses_mastoids'])}<br><br>
            <b>Orbits:</b> {g(['findings', 'orbits'])}
            </p>

            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #007acc; margin-top: 20px;">
                <h3 style="margin-top: 0;">IMPRESSION</h3>
                <ol style="padding-left: 20px;">
        """
        
        impressions = g(['impression'], [])
        if isinstance(impressions, list):
            for imp in impressions:
                html += f"<li>{imp}</li>"
        else:
             html += f"<li>{impressions}</li>"

        html += f"""
                </ol>
            </div>
            
            <div style="margin-top: 40px; font-size: 0.9em; color: #555; text-align: right; border-top: 1px solid #ddd; padding-top: 10px;">
                Electronically Signed by AI Radiology Assistant<br>
                Workstation ID: GEMINI-CLI-01<br>
                Printed on: {print_ts}
            </div>
        </div>
        """
        return html

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

    def _create_llava_prompt(self, prediction, confidence_str, patient_info_formatted, report_date):
        # JSON Template Structure
        json_structure = """
{
  "report_header": {
    "exam_type": "MRI Brain w/ + w/o Contrast",
    "final_report_status": "Final Report"
  },
  "clinical_information": {
    "reason_for_exam": "string",
    "clinical_history": "string (include codes if applicable)",
    "comparison": "string (or 'None')"
  },
  "procedure_details": {
    "examination": "MR OF THE BRAIN WITH AND WITHOUT CONTRAST",
    "technique": "Multisequence, multiplanar imaging...",
    "contrast_info": { "agent": "string", "volume": "string" },
    "dose_info": "string (if applicable)"
  },
  "findings": {
    "cerebral_parenchyma": "string",
    "extra_axial_spaces": "string",
    "ventricles": "string",
    "mass_effect": "string",
    "vascular_structures": "string",
    "bones_and_soft_tissues": "string",
    "paranasal_sinuses_mastoids": "string",
    "orbits": "string"
  },
  "impression": [
    "string (numbered point 1)",
    "string (numbered point 2)"
  ]
}
"""
        
        # Base Prompt
        base_prompt = (
            "You are an expert Radiologist. "
            "Analyze the provided MRI image and generate a full radiology report. "
            "You MUST return the report strictly in valid JSON format corresponding to the following structure. "
            "Do not include any conversational text outside the JSON object.\n\n"
            f"**JSON Structure:**\n```json\n{json_structure}\n```\n\n"
        )

        # Context-Specific Instructions
        if prediction == "Normal" or prediction == "notumor" or prediction == "NonDemented":
            context = (
                f"**Clinical Context:** Patient scanned for routine checkup. AI Classifier predicts: **Normal/No Tumor** ({confidence_str}).\n"
                "**Instructions:**\n"
                "- Fill 'findings' with 'Unremarkable', 'Normal', or 'Preserved' where appropriate.\n"
                "- Ensure 'impression' states 'No acute intracranial abnormality'.\n"
            )
        elif prediction in ALZ_CLASSES:
            formatted_cond = prediction.replace("VeryMild", "Very Mild ").replace("Mild", "Mild ").replace("Moderate", "Moderate ")
            context = (
                f"**Clinical Context:** Evaluation for cognitive decline. AI Classifier predicts: **{formatted_cond} Alzheimer's** ({confidence_str}).\n"
                "**Instructions:**\n"
                "- Focus 'findings' on cortical atrophy, hippocampal volume, and ventricular enlargement.\n"
                "- In 'impression', explicitly mention findings suggestive of dementia/Alzheimer's.\n"
            )
        else: # Tumor
            tumor_name = f"{prediction.capitalize()} Tumor"
            context = (
                f"**Clinical Context:** Patient with suspected mass. AI Classifier predicts: **{tumor_name}** ({confidence_str}).\n"
                "**Instructions:**\n"
                "- In 'cerebral_parenchyma', describe the mass (approximate size, location, signal intensity).\n"
                "- Describe 'mass_effect' (e.g., midline shift, compression).\n"
                "- In 'impression', state the diagnosis of the specific tumor type.\n"
            )

        return base_prompt + context

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