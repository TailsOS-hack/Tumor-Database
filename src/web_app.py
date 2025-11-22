import os
import io
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision import transforms
from PIL import Image
import ollama
import markdown
from weasyprint import HTML as WeasyHTML
from pypdf import PdfWriter, PdfReader

# --- Configuration ---
DEFAULT_MODEL_PATH = os.path.join("models", "brain_tumor_classifier.pt")
app = FastAPI(title="Radiology Report Generator")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# --- Global State (Model) ---
model = None
device = None
class_names = []
eval_tfms = None

# --- Model Helpers ---
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

@app.on_event("startup")
async def load_model_startup():
    global model, device, class_names, eval_tfms
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml as dml
            device = dml.device()
        except ImportError:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Transform setup
    eval_tfms = get_eval_transform()

    # Model loading
    if os.path.isfile(DEFAULT_MODEL_PATH):
        try:
            checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location=device)
            arch = checkpoint.get("arch", "efficientnet_b3")
            class_names = checkpoint.get("class_names", [])
            num_classes = len(class_names)
            
            loaded_model = build_model(arch, num_classes)
            loaded_model.load_state_dict(checkpoint["model_state"])
            loaded_model.eval().to(device)
            
            model = loaded_model
            print(f"Model loaded from {DEFAULT_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found at {DEFAULT_MODEL_PATH}")

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    global model, device, class_names, eval_tfms
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        tensor = eval_tfms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Handle mixed precision if needed, but simple inference is fine
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        top_prob, top_idx = torch.topk(probs, 1)
        pred_label = class_names[top_idx.item()]
        pred_conf = top_prob.item() * 100.0
        
        return {
            "prediction": pred_label,
            "confidence": pred_conf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ReportRequest(BaseModel):
    patient_name: str
    patient_id: str
    dob: str
    tumor_type: str
    confidence: float
    image_base64: str # Base64 encoded image

@app.post("/api/generate_report")
async def generate_report(
    patient_name: str = Form(...),
    patient_id: str = Form(...),
    dob: str = Form(...),
    tumor_type: str = Form(...),
    confidence: float = Form(...),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        
        # Format data for prompt
        confidence_str = f"{confidence:.1f}%"
        report_date = datetime.now().strftime("%B %d, %Y")
        patient_info_formatted = (
            f"**Name:** {patient_name}<br>"
            f"**DOB:** {dob}<br>"
            f"**Patient ID:** {patient_id}"
        )
        
        # Create Prompt
        if tumor_type == "notumor":
            tumor_name = "No Tumor"
            prompt = (
                f"You are a radiologist. A classifier has determined with {confidence_str} confidence that this MRI scan shows **No Tumor**. "
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

        # Call Ollama
        response = ollama.chat(
            model='llava:7b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }],
        )
        
        report_markdown = response['message']['content']
        report_html = markdown.markdown(report_markdown, extensions=['fenced_code', 'tables'])
        
        return JSONResponse(content={"report_html": report_html, "report_markdown": report_markdown})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_pdf")
async def generate_pdf(
    report_markdown: str = Form(...),
    password: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        encoded_string = base64.b64encode(image_bytes).decode('utf-8')
        image_data_uri = f"data:image/jpeg;base64,{encoded_string}"
        
        # Convert markdown to HTML
        pdf_html = markdown.markdown(report_markdown, extensions=['fenced_code', 'tables'])
        
        # Append image
        image_html = f'<div style="text-align: center; margin-top: 20px;"><img src="{image_data_uri}" alt="MRI Scan" style="max-width: 300px; height: auto;"></div>'
        pdf_html += image_html
        
        # Add styles
        css = "body { font-family: sans-serif; font-size: 10pt; } h1, h2, h3 { color: #333; } table { border-collapse: collapse; width: 100%; } td, th { padding: 4px; text-align: left; }"
        html_with_style = f"<style>{css}</style>{pdf_html}"
        
        # Generate PDF
        pdf_buffer = io.BytesIO()
        WeasyHTML(string=html_with_style).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Encrypt if password provided
        if password:
            reader = PdfReader(pdf_buffer)
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.encrypt(password)
            
            encrypted_buffer = io.BytesIO()
            writer.write(encrypted_buffer)
            encrypted_buffer.seek(0)
            pdf_buffer = encrypted_buffer

        headers = {
            'Content-Disposition': 'attachment; filename="radiology_report.pdf"'
        }
        return StreamingResponse(pdf_buffer, headers=headers, media_type='application/pdf')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


