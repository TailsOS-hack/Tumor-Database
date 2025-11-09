import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms

try:
    from PIL import Image, ImageTk
except Exception as e:  # pragma: no cover
    Image = None
    ImageTk = None


APP_TITLE = "Brain Tumor MRI Classifier"
DEFAULT_MODEL_PATH = os.path.join("..", "models", "brain_tumor_classifier.pt")


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
    # ImageNet normalization constants (match EfficientNet pretraining)
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
        super().__init__(master, padding=12)
        self.pack(fill="both", expand=True)

        # Prefer CUDA, fallback to DirectML, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_text = "Device: cuda (GPU detected)"
        else:
            try:
                import torch_directml as dml  # type: ignore
                self.device = dml.device()
                self.device_text = "Device: directml (GPU via DirectML)"
            except Exception:
                self.device = torch.device("cpu")
                self.device_text = "Device: cpu"
        self.model = None
        self.class_names = []
        self.arch = "efficientnet_b3"
        self.image_path = None
        self.tk_image = None
        self.eval_tfms = get_eval_transform()

        self._build_ui()
        # Try auto-load
        if os.path.isfile(DEFAULT_MODEL_PATH):
            self.load_model(DEFAULT_MODEL_PATH)

    def _build_ui(self):
        # Header
        title = ttk.Label(self, text=APP_TITLE, font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        # Top controls
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=4)

        self.model_label = ttk.Label(controls, text=f"Model: (none loaded)")
        self.model_label.pack(side="left")

        ttk.Button(controls, text="Load Model", command=self.on_load_model).pack(
            side="right"
        )

        # Device label
        self.device_label = ttk.Label(self, text=self.device_text)
        self.device_label.pack(anchor="w", pady=(0, 8))

        # Main area: left image, right predictions
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # Image area
        img_frame = ttk.LabelFrame(main, text="Selected Image", padding=8)
        img_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.canvas = tk.Canvas(img_frame, width=380, height=380, bg="#222")
        self.canvas.pack(fill="both", expand=True)

        # Prediction area
        pred_frame = ttk.LabelFrame(main, text="Prediction", padding=8)
        pred_frame.pack(side="right", fill="both", expand=True)

        btn_row = ttk.Frame(pred_frame)
        btn_row.pack(fill="x", pady=(0, 6))

        ttk.Button(btn_row, text="Choose Image", command=self.on_choose_image).pack(
            side="left"
        )
        self.predict_btn = ttk.Button(
            btn_row, text="Classify", command=self.on_predict, state="disabled"
        )
        self.predict_btn.pack(side="right")

        self.pred_label = ttk.Label(
            pred_frame, text="Prediction: -", font=("Segoe UI", 12, "bold")
        )
        self.pred_label.pack(anchor="w", pady=(8, 6))

        # Top-k bars
        self.bars = []
        for _ in range(4):
            row = ttk.Frame(pred_frame)
            row.pack(fill="x", pady=3)
            name = ttk.Label(row, text="-", width=14)
            name.pack(side="left")
            pb = ttk.Progressbar(row, orient="horizontal", length=220, mode="determinate")
            pb.pack(side="left", padx=(6, 0))
            val = ttk.Label(row, text="0.0%", width=8)
            val.pack(side="left", padx=(6, 0))
            self.bars.append((name, pb, val))

        # Footer note
        note = ttk.Label(
            self,
            text="Load the trained model, pick an image, then Classify.",
            foreground="#666",
        )
        note.pack(anchor="w", pady=(8, 0))

    def on_load_model(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[["PyTorch checkpoint", "*.pt;*.pth"], ["All files", "*.*"]],
            initialdir=os.path.abspath("."),
        )
        if not path:
            return
        self.load_model(path)

    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            arch = checkpoint.get("arch", "efficientnet_b3")
            class_names = checkpoint.get("class_names")
            num_classes = checkpoint.get("num_classes", len(class_names) if class_names else 4)
            model = build_model(arch, num_classes)
            model.load_state_dict(checkpoint["model_state"])
            model.eval().to(self.device)

            self.model = model
            self.class_names = class_names or ["glioma", "meningioma", "notumor", "pituitary"]
            self.arch = arch
            self.model_label.configure(text=f"Model: {os.path.basename(path)} ({arch})")
            self.predict_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Load Model", f"Failed to load model:\n{e}")

    def on_choose_image(self):
        if Image is None:
            messagebox.showerror(
                "Pillow not installed",
                "This app requires the 'Pillow' package to display and process images.\n"
                "Install it with: pip install pillow",
            )
            return
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[["Image files", "*.jpg;*.jpeg;*.png;*.bmp"], ["All files", "*.*"]],
            initialdir=os.path.abspath("."),
        )
        if not path:
            return
        self.image_path = path
        self._display_image(path)

    def _display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            # Fit to canvas
            canvas_w = int(self.canvas["width"]) or 380
            canvas_h = int(self.canvas["height"]) or 380
            img.thumbnail((canvas_w, canvas_h))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image)
        except Exception as e:
            messagebox.showerror("Image", f"Failed to load image:\n{e}")

    @torch.no_grad()
    def on_predict(self):
        if self.model is None:
            messagebox.showinfo("Model", "Load a model first.")
            return
        if not self.image_path:
            messagebox.showinfo("Image", "Choose an image to classify.")
            return
        try:
            img = Image.open(self.image_path).convert("RGB")
            tensor = self.eval_tfms(img).unsqueeze(0)
            tensor = tensor.to(self.device)
            # Mixed precision on CUDA only
            use_amp = False
            try:
                use_amp = (hasattr(self.device, "type") and self.device.type == "cuda")
            except Exception:
                use_amp = False
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            probs_list = probs.detach().cpu().tolist()

            # Top-k
            topk = min(4, len(self.class_names))
            top_probs, top_idxs = torch.topk(probs, k=topk)
            top_probs = top_probs.cpu().tolist()
            top_idxs = top_idxs.cpu().tolist()
            top_labels = [self.class_names[i] for i in top_idxs]

            pred_label = top_labels[0]
            pred_conf = top_probs[0] * 100.0
            self.pred_label.configure(
                text=f"Prediction: {pred_label} ({pred_conf:.1f}%)"
            )

            # Update bars
            for i, (name, pb, val) in enumerate(self.bars):
                if i < topk:
                    name.configure(text=top_labels[i])
                    pb.configure(value=int(top_probs[i] * 100))
                    val.configure(text=f"{top_probs[i]*100:.1f}%")
                else:
                    name.configure(text="-")
                    pb.configure(value=0)
                    val.configure(text="0.0%")
        except Exception as e:
            messagebox.showerror("Predict", f"Failed to classify image:\n{e}")


def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    try:
        style = ttk.Style()
        theme = "vista" if sys.platform.startswith("win") else "clam"
        style.theme_use(theme)
    except Exception:
        pass
    root.geometry("820x520")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
