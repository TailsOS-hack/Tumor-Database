# %% [markdown]
# # Multimodal LLM Benchmark + LoRA Fine-Tuning
#
# Run this file in Google Colab with an A100 runtime when possible. It keeps
# model downloads, multimodal inference, and adapter training off the MacBook.

# %% [markdown]
# ## 1. Install dependencies

# %%
# In Colab, uncomment and run:
# !pip install -q --upgrade "transformers>=4.49.0" accelerate bitsandbytes peft trl datasets pillow pandas scikit-learn qwen-vl-utils

# %% [markdown]
# ## 2. Mount Drive and configure paths

# %%
from pathlib import Path
import json
import os
import re

import pandas as pd
from PIL import Image

# In Colab, uncomment:
# from google.colab import drive
# drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("TUMOR_DB_DRIVE_ROOT", "/content/drive/MyDrive/Tumor-Database"))
REPO_ROOT = Path(os.environ.get("TUMOR_DB_REPO_ROOT", str(DRIVE_ROOT)))
MANIFEST_PATH = Path(
    os.environ.get(
        "TUMOR_DB_MANIFEST",
        str(REPO_ROOT / "training_logs" / "splits" / "strict_manifest.csv"),
    )
)
IMAGE_ROOT = Path(os.environ.get("TUMOR_DB_IMAGE_ROOT", str(REPO_ROOT)))
RESULTS_DIR = Path(os.environ.get("TUMOR_DB_RESULTS_DIR", str(DRIVE_ROOT / "multimodal_results")))
ADAPTER_DIR = Path(os.environ.get("TUMOR_DB_ADAPTER_DIR", str(DRIVE_ROOT / "lora_adapters")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "tumor_glioma",
    "tumor_meningioma",
    "tumor_notumor",
    "tumor_pituitary",
    "dementia_MildDemented",
    "dementia_ModerateDemented",
    "dementia_NonDemented",
    "dementia_VeryMildDemented",
]

MODEL_CANDIDATES = [
    {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "notes": "Strong first pass; supports structured outputs; manageable on A100.",
        "trust_remote_code": False,
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "notes": "Larger Qwen candidate for A100 4-bit runs.",
        "trust_remote_code": False,
    },
    {
        "model_id": "llava-hf/llava-v1.6-34b-hf",
        "notes": "Requested LLaVA-34B baseline; use 4-bit on A100.",
        "trust_remote_code": False,
    },
    {
        "model_id": "microsoft/Phi-3.5-vision-instruct",
        "notes": "Smaller vision model; useful latency/quality comparison.",
        "trust_remote_code": True,
    },
    {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "notes": "Gated model; requires accepted Meta license and HF token.",
        "trust_remote_code": False,
    },
]


# %% [markdown]
# ## 3. Load strict test examples

# %%
def load_eval_rows(limit_per_class=20):
    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest[manifest["split"] == "test"].copy()
    chunks = []
    for label, group in test_rows.groupby("eight_class"):
        chunks.append(group.sort_values("path").head(limit_per_class))
    return pd.concat(chunks, ignore_index=True)


eval_rows = load_eval_rows(limit_per_class=20)
eval_rows.head()


# %% [markdown]
# ## 4. Prompt and parser

# %%
SYSTEM_PROMPT = """You are evaluating a brain MRI image for a research benchmark.
Return only strict JSON. Do not include markdown.
The label must be exactly one of the allowed labels."""


def build_prompt():
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Classify this MRI image. Allowed labels: "
                    + ", ".join(LABELS)
                    + '. Return JSON like {"label":"...", "confidence":0.0, "rationale":"one short sentence"}.'
                ),
            }
        ],
    }


def parse_json_response(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "rationale": text[:200]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "rationale": text[:200]}
    label = data.get("label", "PARSE_ERROR")
    if label not in LABELS:
        label = "INVALID_LABEL"
    return {
        "label": label,
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "rationale": str(data.get("rationale", ""))[:500],
    }


# %% [markdown]
# ## 5. Generic Transformers inference runner

# %%
import torch
from transformers import BitsAndBytesConfig, pipeline


def load_pipe(model_id, trust_remote_code=False, load_in_4bit=True):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": trust_remote_code,
        },
    )


def run_model(candidate, rows, max_new_tokens=128):
    model_id = candidate["model_id"]
    pipe = load_pipe(
        model_id,
        trust_remote_code=candidate.get("trust_remote_code", False),
        load_in_4bit=True,
    )

    outputs = []
    for _, row in rows.iterrows():
        image_path = IMAGE_ROOT / row["path"]
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            build_prompt(),
        ]
        messages[1]["content"].insert(0, {"type": "image", "image": image})
        raw = pipe(text=messages, max_new_tokens=max_new_tokens)
        generated = raw[0].get("generated_text", raw[0])
        if isinstance(generated, list):
            generated = generated[-1].get("content", "")
        parsed = parse_json_response(str(generated))
        outputs.append(
            {
                "model_id": model_id,
                "path": row["path"],
                "true_label": row["eight_class"],
                "pred_label": parsed["label"],
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
                "correct": parsed["label"] == row["eight_class"],
            }
        )

    result_df = pd.DataFrame(outputs)
    safe_name = model_id.replace("/", "__")
    result_df.to_csv(RESULTS_DIR / f"{safe_name}_eval.csv", index=False)
    summary = {
        "model_id": model_id,
        "n": len(result_df),
        "accuracy": float(result_df["correct"].mean()) if len(result_df) else 0.0,
        "strict_json_rate": float((~result_df["pred_label"].isin(["PARSE_ERROR", "INVALID_LABEL"])).mean())
        if len(result_df)
        else 0.0,
    }
    (RESULTS_DIR / f"{safe_name}_summary.json").write_text(json.dumps(summary, indent=2))
    del pipe
    torch.cuda.empty_cache()
    return summary


# %% [markdown]
# ## 6. Run 3-5 candidates

# %%
# summaries = []
# for candidate in MODEL_CANDIDATES[:5]:
#     print("Running", candidate["model_id"])
#     summaries.append(run_model(candidate, eval_rows, max_new_tokens=128))
# pd.DataFrame(summaries).sort_values("accuracy", ascending=False)


# %% [markdown]
# ## 7. Prepare LoRA training data
#
# Create a JSONL file with rows shaped like:
#
# `{"image":"data/evaluation/images/example.jpg","label":"tumor_glioma","answer_json":{"label":"tumor_glioma","confidence":1.0,"rationale":"..."} }`

# %%
TRAIN_JSONL = DRIVE_ROOT / "multimodal_lora_train.jsonl"
VAL_JSONL = DRIVE_ROOT / "multimodal_lora_val.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


# %% [markdown]
# ## 8. LoRA fine-tuning skeleton
#
# Start with the best benchmark model. For first fine-tuning, prefer Qwen/Qwen2.5-VL-7B-Instruct because it is easier to iterate than 32B/34B checkpoints.

# %%
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
from trl import SFTTrainer


def format_lora_example(example):
    answer = json.dumps(example["answer_json"], ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(IMAGE_ROOT / example["image"])},
                    {"type": "text", "text": build_prompt()["content"][0]["text"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def train_lora(base_model_id="Qwen/Qwen2.5-VL-7B-Instruct", adapter_name="qwen25vl_7b_mri_lora"):
    train_data = Dataset.from_list([format_lora_example(row) for row in load_jsonl(TRAIN_JSONL)])
    val_data = Dataset.from_list([format_lora_example(row) for row in load_jsonl(VAL_JSONL)])

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR / adapter_name),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR / adapter_name))
    processor.save_pretrained(str(ADAPTER_DIR / adapter_name))
    return ADAPTER_DIR / adapter_name


# %%
# adapter_path = train_lora()
# adapter_path
