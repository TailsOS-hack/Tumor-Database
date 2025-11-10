
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
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
from tkcalendar import DateEntry
import io
from pypdf import PdfWriter, PdfReader
import base64

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
        self.last_report_markdown = None

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
        self.report_html.set_html("<p><i>Generating report with multimodal LLM, please wait... This may take a moment.</i></p>")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start()
        self.update_idletasks()

        try:
            pred_label, pred_conf = self._run_classification()
            self.pred_label.configure(text=f"Prediction: {pred_label}")
            self.confidence_label.configure(text=f"Confidence: {pred_conf:.2f}%")
            
            # Pre-format confidence and patient info for the LLM
            confidence_str = f"{pred_conf:.1f}%"
            patient_info_formatted = (
                f"**Name:** {self.patient_name_entry.get()}<br>"
                f"**DOB:** {self.dob_entry.get_date().strftime('%Y-%m-%d')}<br>"
                f"**Patient ID:** {self.patient_id_entry.get()}"
            )
            self._start_report_generation(pred_label, confidence_str, patient_info_formatted)

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
        
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        
        top_prob, top_idx = torch.topk(probs, 1)
        pred_label = self.class_names[top_idx.item()]
        pred_conf = top_prob.item() * 100.0
        return pred_label, pred_conf

    def _start_report_generation(self, tumor_type, confidence_str, patient_info_formatted):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(tumor_type, confidence_str, patient_info_formatted),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, tumor_type, confidence_str, patient_info_formatted):
        try:
            report_date = datetime.now().strftime("%B %d, %Y")
            
            image_bytes = None
            try:
                with open(self.image_path, "rb") as image_file:
                    image_bytes = image_file.read()
            except Exception as e:
                print(f"Could not read image for LLM: {e}")

            prompt = self._create_llava_prompt(tumor_type, confidence_str, patient_info_formatted, report_date)
            
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
        if not self.last_report_markdown:
            messagebox.showwarning("No Report", "Please generate a report before saving.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        password = simpledialog.askstring("PDF Encryption", "Enter a password to encrypt the PDF (optional):", show='*')

        try:
            # Convert the main report body to HTML
            pdf_html = markdown.markdown(self.last_report_markdown, extensions=['fenced_code', 'tables'])

            # Now, create and append the image HTML programmatically
            image_data_uri = None
            try:
                with open(self.image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data_uri = f"data:image/jpeg;base64,{encoded_string}"
            except Exception as e:
                print(f"Could not read or encode image for PDF: {e}")

            if image_data_uri:
                image_html = f'<div style="text-align: center; margin-top: 20px;"><img src="{image_data_uri}" alt="MRI Scan" style="max-width: 300px; height: auto;"></div>'
                pdf_html += image_html
            else:
                pdf_html += "<p><i>[Image could not be loaded]</i></p>"

            css = "body { font-family: sans-serif; font-size: 10pt; } h1, h2, h3 { color: #333; } table { border-collapse: collapse; width: 100%; } td, th { padding: 4px; text-align: left; }"
            html_with_style = f"<style>{css}</style>{pdf_html}"
            
            pdf_buffer = io.BytesIO()
            WeasyHTML(string=html_with_style, base_url=self.image_path).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)

            if password:
                reader = PdfReader(pdf_buffer)
                writer = PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                writer.encrypt(password)
                with open(filepath, "wb") as f:
                    writer.write(f)
                messagebox.showinfo("Success", f"Encrypted report saved successfully to:\n{filepath}")
            else:
                with open(filepath, "wb") as f:
                    f.write(pdf_buffer.read())
                messagebox.showinfo("Success", f"Report saved successfully to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("PDF Export Error", f"Failed to save PDF:\n{e}")

    def _create_llava_prompt(self, tumor_type, confidence_str, patient_info_formatted, report_date):
        if tumor_type == "notumor":
            tumor_name = "No Tumor"
            prompt = (
                "You are a radiologist. A classifier has determined with {confidence_str} confidence that this MRI scan shows **No Tumor**. "
                "Your task is to write a concise, formal radiology report confirming this finding. "
                "Use the provided template and data. Do not add extra notes or disclaimers.\n\n"
                "### CRITICAL FORMATTING INSTRUCTIONS ###\n"
                "1.  **Bold Headers:** Make the 'Patient Details', 'FINDINGS', and 'IMPRESSION' headers bold using markdown (`**Header**`).\n\n"
                "--- TEMPLATE FOR OUTPUT ---\n"
                '<div style="text-align: center;"><h1>RADIOLOGY REPORT</h1></div>\n\n'
                "**Patient Details:**<br>"
                f"{patient_info_formatted}<br><br>"
                f"**Date of Report:** {report_date}<br><br>"
                "**FINDINGS:**<br>"
                f"- An AI model analyzed the images and identified features consistent with **{tumor_name}** (Confidence: {confidence_str}).<br>"
                "- The scan shows no evidence of an intracranial mass, lesion, or other significant abnormality.<br><br>"
                "**IMPRESSION:**<br>"
                f"- No evidence of intracranial tumor."
            )
        else:
            tumor_name = f"{tumor_type.capitalize()} Tumor"
            prompt = (
                f"You are a radiologist. A highly accurate classifier has already identified a **{tumor_name}** in this MRI scan with {confidence_str} confidence. "
                "Your task is to analyze the provided image and write a formal radiology report.\n\n"
                "### CRITICAL INSTRUCTIONS ###\n"
                "1.  **Analyze the Image:** Look at the tumor in the image.\n"
                "2.  **Estimate Size:** Based on the image, estimate the tumor's dimensions in millimeters (mm). State this as 'approx. X x Y mm'.\n"
                "3.  **Describe Appearance:** Describe the specific appearance of the tumor you see in the image (e.g., its shape, margins, signal intensity, location). Do NOT use generic or 'typical' descriptions.\n"
                "4.  **Format Headers:** Make the 'Patient Details', 'FINDINGS', and 'IMPRESSION' headers bold using markdown (`**Header**`).\n"
                "5.  **Use Template:** Format your entire response using the template below. Do not add extra notes or disclaimers.\n\n"
                "--- TEMPLATE FOR OUTPUT ---\n"
                '<div style="text-align: center;"><h1>RADIOLOGY REPORT</h1></div>\n\n'
                "**Patient Details:**<br>"
                f"{patient_info_formatted}<br><br>"
                f"**Date of Report:** {report_date}<br><br>"
                "**FINDINGS:**<br>"
                f"- An AI model identified features consistent with a **{tumor_name}** (Confidence: {confidence_str}).<br>"
                "- The lesion is estimated to have approximate dimensions of [**INSERT YOUR ESTIMATED DIMENSIONS HERE**].<br>"
                "- The scan reveals [**INSERT YOUR DESCRIPTION OF THE TUMOR'S APPEARANCE HERE**].<br><br>"
                "**IMPRESSION:**<br>"
                f"- The findings are most consistent with a **{tumor_name}**."
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
