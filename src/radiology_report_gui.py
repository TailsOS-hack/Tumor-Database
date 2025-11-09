
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision import transforms
import ollama
from PIL import Image, ImageTk
import threading
import queue
import markdown
from tkhtmlview import HTMLLabel
from datetime import datetime
from weasyprint import HTML as WeasyHTML

APP_TITLE = "Radiology Report Generator"
DEFAULT_MODEL_PATH = os.path.join("models", "brain_tumor_classifier.pt")

def build_model(arch: str, num_classes: int):
    if arch == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def get_eval_transform():
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

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill="both", expand=True)
        self.master = master

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            try:
                import torch_directml as dml
                self.device = dml.device()
            except Exception:
                self.device = torch.device("cpu")

        self.model = None
        self.class_names = []
        self.arch = "efficientnet_b3"
        self.image_path = None
        self.tk_image = None
        self.eval_tfms = get_eval_transform()
        self.report_queue = queue.Queue()
        self.last_report_html = None

        self._build_ui()
        if os.path.isfile(DEFAULT_MODEL_PATH):
            self.load_model(DEFAULT_MODEL_PATH)

    def _build_ui(self):
        # Top bar for model status and loading
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", pady=(0, 10))
        self.model_label = ttk.Label(top_bar, text="Model: (none loaded)", anchor="w")
        self.model_label.pack(side="left", fill="x", expand=True)
        ttk.Button(top_bar, text="Load Classifier Model", command=self.on_load_model).pack(side="right")

        # Main layout with a resizable pane
        main_pane = ttk.PanedWindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        # --- Left Column: Image and Classification ---
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        # Patient Info
        patient_frame = ttk.LabelFrame(left_frame, text="Patient Information", padding=10)
        patient_frame.pack(fill="x", pady=(0, 10))
        self.patient_info_text = tk.Text(patient_frame, height=4, font=("Segoe UI", 9))
        self.patient_info_text.pack(fill="x", expand=True)
        self.patient_info_text.insert("1.0", "Name: \nDOB: \nPatient ID: ")

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
        self.progress_bar.pack_forget() # Hide it initially

        # Classification result
        result_frame = ttk.LabelFrame(left_frame, text="Classification Details", padding=10)
        result_frame.pack(fill="x")
        self.pred_label = ttk.Label(result_frame, text="Prediction: -", font=("Segoe UI", 11, "bold"))
        self.pred_label.pack(anchor="w")
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -")
        self.confidence_label.pack(anchor="w")

        # --- Right Column: Generated Report ---
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=2)
        
        report_frame = ttk.LabelFrame(right_frame, text="Generated Radiology Report", padding=10)
        report_frame.pack(fill="both", expand=True)
        
        self.report_html = HTMLLabel(report_frame, html="<p>Report will be generated here.</p>")
        self.report_html.pack(fill="both", expand=True)
        
        self.save_pdf_btn = ttk.Button(right_frame, text="Save as PDF", command=self.on_save_pdf, state="disabled")
        self.save_pdf_btn.pack(pady=5)

    def on_load_model(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[["PyTorch checkpoint", "*.pt;*.pth"]],
            initialdir=os.path.abspath("models"),
        )
        if path:
            self.load_model(path)

    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.arch = checkpoint.get("arch", "efficientnet_b3")
            self.class_names = checkpoint.get("class_names", [])
            num_classes = len(self.class_names)
            
            model = build_model(self.arch, num_classes)
            model.load_state_dict(checkpoint["model_state"])
            model.eval().to(self.device)

            self.model = model
            self.model_label.configure(text=f"Model: {os.path.basename(path)} ({self.arch} on {self.device})")
            if self.image_path:
                self.generate_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Load Model Error", f"Failed to load model:\n{e}")

    def on_choose_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[["Image files", "*.jpg;*.jpeg;*.png;*.bmp"]],
            initialdir=os.path.abspath("data"),
        )
        if path:
            self.image_path = path
            self._display_image(path)
            if self.model:
                self.generate_btn.configure(state="normal")

    def _display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w == 1 or h == 1: # Canvas not yet drawn
                w, h = 350, 350
            img.thumbnail((w, h))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self.tk_image)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

    def on_analyze_and_generate(self):
        if not self.image_path or not self.model:
            return

        self.generate_btn.configure(state="disabled")
        self.choose_img_btn.configure(state="disabled")
        self.save_pdf_btn.configure(state="disabled")
        self.pred_label.configure(text="Prediction: Analyzing...")
        self.confidence_label.configure(text="Confidence: -")
        self.report_html.set_html("<p><i>Generating report, please wait...</i></p>")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start()
        self.update_idletasks()

        try:
            pred_label, pred_conf = self._run_classification()
            self.pred_label.configure(text=f"Prediction: {pred_label}")
            self.confidence_label.configure(text=f"Confidence: {pred_conf:.2f}%")
            
            patient_info = self.patient_info_text.get("1.0", tk.END)
            self._start_report_generation(pred_label, pred_conf, patient_info)

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{e}")
            self.generate_btn.configure(state="normal")
            self.choose_img_btn.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

    @torch.no_grad()
    def _run_classification(self):
        img = Image.open(self.image_path).convert("RGB")
        tensor = self.eval_tfms(img).unsqueeze(0).to(self.device)
        
        # Use the recommended torch.amp.autocast
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        
        top_prob, top_idx = torch.topk(probs, 1)
        pred_label = self.class_names[top_idx.item()]
        pred_conf = top_prob.item() * 100.0
        return pred_label, pred_conf

    def _start_report_generation(self, tumor_type, confidence, patient_info):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(tumor_type, confidence, patient_info),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, tumor_type, confidence, patient_info):
        try:
            report_date = datetime.now().strftime("%B %d, %Y")
            prompt = self._create_llm_prompt(tumor_type, confidence, patient_info, report_date)
            
            response = ollama.chat(
                model='llama3.2:3b',
                messages=[{'role': 'user', 'content': prompt}],
            )
            
            markdown_text = response['message']['content']
            html_text = markdown.markdown(markdown_text, extensions=['fenced_code', 'tables'])
            self.report_queue.put(html_text)

        except Exception as e:
            error_html = f"<p><b>Error generating report:</b></p><pre>{e}</pre>"
            self.report_queue.put(error_html)

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
        if not self.last_report_html:
            messagebox.showwarning("No Report", "Please generate a report before saving.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        try:
            # Basic CSS for better PDF formatting
            css = """
            body { font-family: sans-serif; font-size: 10pt; }
            h1, h2, h3 { color: #333; }
            h3 { margin-top: 2em; margin-bottom: 0.5em; }
            p { margin-top: 0; }
            pre { background-color: #f0f0f0; padding: 1em; border: 1px solid #ddd; }
            """
            html_with_style = f"<style>{css}</style>{self.last_report_html}"
            WeasyHTML(string=html_with_style).write_pdf(filepath)
            messagebox.showinfo("Success", f"Report saved successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("PDF Export Error", f"Failed to save PDF:\n{e}")

    def _create_llm_prompt(self, tumor_type, confidence, patient_info, report_date):
        uncertainty_threshold = 85.0
        
        prompt = (
            "You are a board-certified radiologist generating a formal report. "
            "An AI model has provided a preliminary classification. Your task is to format this into a complete clinical document.\n\n"
            "### DO NOT invent any specific details like measurements, slice numbers, or precise locations. ###\n\n"
            "Use the following template and information. Fill in the 'FINDINGS' and 'IMPRESSION' sections based on the AI's output.\n\n"
            "--- TEMPLATE START ---\n"
            "### RADIOLOGY REPORT\n\n"
            f"**Patient Details:**\n{patient_info}\n\n"
            f"**Date of Report:** {report_date}\n\n"
            "**History:** Routine brain scan for analysis.\n\n"
            "**Technique:** Multi-planar, multi-sequence MRI of the brain was performed without intravenous contrast.\n\n"
            f"**AI-Assisted Findings:**\nAn AI model analyzed the images and identified features consistent with **{tumor_type}** (Confidence: {confidence:.1f}%).\n\n"
            "### FINDINGS:\n"
            "- Based on the AI finding, describe the general, textbook imaging characteristics of a '{tumor_type}'.\n"
        )

        if confidence < uncertainty_threshold:
            prompt += (
                "- The imaging characteristics are suggestive, but not entirely conclusive due to the model's confidence level. Further clinical correlation is advised.\n"
            )
        
        prompt += (
            "\n### IMPRESSION:\n"
            "- State the most likely diagnosis based on the AI finding.\n\n"
            "--- TEMPLATE END ---\n"
        )
        return prompt

def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1100x750")
    
    try:
        style = ttk.Style()
        if sys.platform.startswith("win"):
            style.theme_use("vista")
        else:
            style.theme_use("clam")
    except Exception:
        pass

    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
