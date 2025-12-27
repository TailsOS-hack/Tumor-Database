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

APP_TITLE = "Radiology Report Generator"
GATEKEEPER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gatekeeper_classifier.pt")
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "brain_tumor_classifier.pt")
ALZHEIMERS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alzheimers_classifier.pt")

# Constants for Alzheimer's
ALZ_CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
ALZ_IMG_SIZE = 160

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
            transforms.Resize(320, antialias=True),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_alzheimers_transform():
    # Matches the MobileNetV3 training (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((ALZ_IMG_SIZE, ALZ_IMG_SIZE), antialias=True),
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

        # Patient Info
        patient_frame = ttk.LabelFrame(left_frame, text="Patient Information", padding=10)
        patient_frame.pack(fill="x", pady=(0, 10))
        
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

        # Image selection and display
        img_frame = ttk.LabelFrame(left_frame, text="MRI Scan", padding=10)
        img_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(img_frame, width=350, height=350, bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.canvas.pack(fill="both", expand=True, pady=(5, 10))
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill="x", pady=10)
        
        self.choose_img_btn = ttk.Button(analysis_frame, text="Choose Image...", command=self.on_choose_image)
        self.choose_img_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.generate_btn = ttk.Button(analysis_frame, text="Analyze & Generate Report", command=self.on_analyze_and_generate, state="disabled")
        self.generate_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Loading indicator
        self.progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.pack_forget()

        # Classification result
        result_frame = ttk.LabelFrame(left_frame, text="Classification Details", padding=10)
        result_frame.pack(fill="x")
        self.pred_label = ttk.Label(result_frame, text="Prediction: -", font=("Segoe UI", 11, "bold"))
        self.pred_label.pack(anchor="w")
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -")
        self.confidence_label.pack(anchor="w")
        self.model_used_label = ttk.Label(result_frame, text="Model Used: -", foreground="#555")
        self.model_used_label.pack(anchor="w")

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
                model = GatekeeperClassifier(freeze_base=False)
                model.load_state_dict(torch.load(GATEKEEPER_MODEL_PATH, map_location=self.device))
                model.eval().to(self.device)
                self.gatekeeper_model = model
                status_texts.append("Gatekeeper: Ready")
            except Exception as e:
                print(f"Error loading gatekeeper model: {e}")
                status_texts.append("Gatekeeper: Error")
        else:
            status_texts.append("Gatekeeper: Not Found")

        # 2. Load Tumor Model
        if os.path.isfile(TUMOR_MODEL_PATH):
            try:
                checkpoint = torch.load(TUMOR_MODEL_PATH, map_location=self.device)
                arch = checkpoint.get("arch", "efficientnet_b3")
                self.tumor_classes = checkpoint.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
                model = build_tumor_model(arch, len(self.tumor_classes))
                model.load_state_dict(checkpoint["model_state"])
                model.eval().to(self.device)
                self.tumor_model = model
                status_texts.append("Tumor: Ready")
            except Exception as e:
                print(f"Error loading tumor model: {e}")
                status_texts.append("Tumor: Error")
        else:
            status_texts.append("Tumor: Not Found")

        # 2. Load Alzheimer's Model
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
            initialdir=os.path.abspath("data"),
        )
        if path:
            self.image_path = path
            self._display_image(path)
            if self.tumor_model or self.alz_model:
                self.generate_btn.configure(state="normal")

    def _display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w == 1 or h == 1: w, h = 350, 350
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
            self._start_report_generation(pred_label, confidence_str, patient_info_formatted)

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
        if self.gatekeeper_model:
            # ResNet50 typically uses 224x224, but let's use the tumor transform for consistency or specific gatekeeper ones
            # Actually gatekeeper was trained with default load_data which uses some transform.
            # Let's use the tumor transform as it's close enough (300x300) and usually works well for MRI
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                output = self.gatekeeper_model(tensor)
                prob = torch.sigmoid(output).item()
            
            # Class 0 = Tumor, Class 1 = Dementia (per train_gatekeeper.py)
            if prob > 0.5:
                target_domain = "dementia"
                print(f"Gatekeeper: Dementia detected (confidence {prob*100:.2f}%)")
            else:
                target_domain = "tumor"
                print(f"Gatekeeper: Tumor scan detected (confidence {(1-prob)*100:.2f}%)")

        # 2. Specialized Classification Phase
        if target_domain == "dementia" and self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                logits = self.alz_model(tensor_alz)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's/Dementia Model"
            
        elif target_domain == "tumor" and self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                logits = self.tumor_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model"
        
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

    def _start_report_generation(self, prediction, confidence_str, patient_info_formatted):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(prediction, confidence_str, patient_info_formatted),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, prediction, confidence_str, patient_info_formatted):
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
            
            self.last_report_markdown = response['message']['content']
            html_text = markdown.markdown(self.last_report_markdown, extensions=['fenced_code', 'tables'])
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
        if not self.last_report_markdown: return

        filepath = filedialog.asksaveasfilename(
            title="Save Report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not filepath: return

        password = simpledialog.askstring("PDF Encryption", "Enter a password (optional):", show='*')

        try:
            pdf_html = markdown.markdown(self.last_report_markdown, extensions=['fenced_code', 'tables'])
            
            # Embed image
            image_data_uri = None
            try:
                with open(self.image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    image_data_uri = f"data:image/jpeg;base64,{encoded}"
            except Exception: pass

            if image_data_uri:
                pdf_html += f'<div style="text-align: center; margin-top: 20px;"><img src="{image_data_uri}" alt="MRI Scan" style="max-width: 300px; height: auto;"></div>'

            css = "body { font-family: sans-serif; font-size: 10pt; } h1, h2, h3 { color: #333; } table { border-collapse: collapse; width: 100%; } td, th { padding: 4px; text-align: left; border-bottom: 1px solid #ddd; }"
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
        # Determine context based on prediction
        if prediction in ALZ_CLASSES:
            # Alzheimer's Context
            cond = prediction
            if cond == "NonDemented":
                finding_text = f"The analysis finds features consistent with a **Non-Demented** (normal) brain (Confidence: {confidence_str})."
                imp_text = "Normal brain MRI study with no significant signs of atrophy or dementia."
                task = "Verify normal brain volume and absence of significant atrophy."
            else:
                formatted_cond = cond.replace("VeryMild", "Very Mild ").replace("Mild", "Mild ").replace("Moderate", "Moderate ")
                finding_text = f"The analysis identifies patterns consistent with **{formatted_cond}** (Confidence: {confidence_str})."
                imp_text = f"Findings suggestive of {formatted_cond}."
                task = "Describe the degree of cortical atrophy and ventricular enlargement visible in the image."
            
            prompt = (
                f"You are a specialized Neuroradiologist. An AI classifier has identified this scan as **{prediction}** (Confidence: {confidence_str}). "
                f"Your task is to write a report focusing on signs of neurodegeneration.\n\n"
                f"**Task:** {task}\n"
                "1. **Analyze the Image:** Look for cortical atrophy, hippocampal volume loss, and ventricular size.\n"
                "--- TEMPLATE ---\n"
                '<div style="text-align: center;"><h1>NEURORADIOLOGY REPORT</h1></div>\n\n'
                "**Patient Details:**<br>"
                f"{patient_info_formatted}<br><br>"
                f"**Date of Report:** {report_date}<br><br>"
                "**FINDINGS:**<br>"
                f"- {finding_text}<br>"
                "- [**DESCRIBE VENTRICLES AND SULCI HERE BASED ON IMAGE**]<br>"
                "- [**DESCRIBE HIPPOCAMPAL/TEMPORAL LOBE APPEARANCE HERE**]<br><br>"
                "**IMPRESSION:**<br>"
                f"- {imp_text}"
            )
            
        elif prediction == "notumor":
            # No Tumor Context
            prompt = (
                f"You are a radiologist. Classifier: **No Tumor** ({confidence_str}). "
                "Write a concise normal report.\n\n"
                "--- TEMPLATE ---\n"
                '<div style="text-align: center;"><h1>RADIOLOGY REPORT</h1></div>\n\n'
                "**Patient Details:**<br>"
                f"{patient_info_formatted}<br><br>"
                f"**Date of Report:** {report_date}<br><br>"
                "**FINDINGS:**<br>"
                f"- AI Analysis: **No Tumor** ({confidence_str}).<br>"
                "- No intracranial mass or acute abnormality.<br><br>"
                "**IMPRESSION:**<br>"
                "- Normal MRI brain."
            )
        else:
            # Tumor Context
            tumor_name = f"{prediction.capitalize()} Tumor"
            prompt = (
                f"You are a radiologist. Classifier: **{tumor_name}** ({confidence_str}). "
                "Write a detailed tumor report.\n\n"
                "1. **Estimate Size:** (approx mm).\n"
                "2. **Describe:** Shape, margins, intensity.\n\n"
                "--- TEMPLATE ---\n"
                '<div style="text-align: center;"><h1>RADIOLOGY REPORT</h1></div>\n\n'
                "**Patient Details:**<br>"
                f"{patient_info_formatted}<br><br>"
                f"**Date of Report:** {report_date}<br><br>"
                "**FINDINGS:**<br>"
                f"- AI Analysis: **{tumor_name}** ({confidence_str}).<br>"
                "- Dimensions: [**ESTIMATE SIZE**]<br>"
                "- Appearance: [**DESCRIBE TUMOR**]<br><br>"
                "**IMPRESSION:**<br>"
                f"- Findings consistent with **{tumor_name}**."
            )
        return prompt

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