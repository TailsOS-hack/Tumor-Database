"""Kaggle fire-and-forget multimodal MRI benchmark + LoRA adapter job.

This script is designed to run as a Kaggle script kernel. It clones the public
Tumor-Database repo, rebuilds the strict manifest, benchmarks feasible open
vision-language models, trains a small Qwen LoRA adapter if resources allow,
and writes every useful artifact to /kaggle/working.

Current batch mode: hierarchical VLM diagnostic. The earlier flat 8-class and
LoRA paths are retained for reproducibility, but the next remote run focuses on
whether a VLM performs better when asked for tumor-vs-dementia first and only
then for the relevant 4-way subtype.
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORK_DIR = Path("/kaggle/working")
REPO_URL = "https://github.com/TailsOS-hack/Tumor-Database.git"
REPO_DIR = WORK_DIR / "Tumor-Database"
OUTPUT_DIR = WORK_DIR / "tumor_multimodal_kaggle_outputs"
PROGRESS_PATH = OUTPUT_DIR / "progress.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "kaggle_multimodal_summary.json"
ZIP_PATH = WORK_DIR / "tumor_multimodal_kaggle_outputs.zip"

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

DOMAIN_LABELS = ["tumor", "dementia"]
TUMOR_SUBTYPES = ["glioma", "meningioma", "notumor", "pituitary"]
DEMENTIA_SUBTYPES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

SYSTEM_PROMPT = (
    "You are classifying brain MRI images for a research benchmark. "
    "This is not clinical diagnosis. Return only strict JSON."
)

DOMAIN_PROMPT = (
    "Classify this brain MRI image into exactly one broad domain: tumor or dementia. "
    'Return only JSON like {"domain":"tumor","confidence":0.75,'
    '"rationale":"short visual reason"} .'
)

TUMOR_SUBTYPE_PROMPT = (
    "The broad domain is tumor. Choose exactly one tumor subtype label: "
    + ", ".join(TUMOR_SUBTYPES)
    + '. Return only JSON like {"subtype":"glioma","confidence":0.75,'
    + '"rationale":"short visual reason"} .'
)

DEMENTIA_SUBTYPE_PROMPT = (
    "The broad domain is dementia. Choose exactly one dementia subtype label: "
    + ", ".join(DEMENTIA_SUBTYPES)
    + '. Return only JSON like {"subtype":"MildDemented","confidence":0.75,'
    + '"rationale":"short visual reason"} .'
)

USER_PROMPT = (
    "Classify this brain MRI image. Choose exactly one allowed label: "
    + ", ".join(LABELS)
    + '. Return only JSON like {"label":"tumor_glioma","confidence":0.75,'
    + '"rationale":"short visual reason"} .'
)

MODEL_CANDIDATES = [
    {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 8,
        "min_gpu_gb": 10,
        "notes": "Primary free-tier model. Strong quality with realistic Kaggle memory needs.",
    },
    {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 6,
        "min_gpu_gb": 10,
        "notes": "Smaller Qwen baseline for free T4 runs.",
    },
    {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 6,
        "min_gpu_gb": 10,
        "notes": "Small open VLM baseline that should fit on Kaggle T4.",
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 5,
        "min_gpu_gb": 24,
        "notes": "Stronger Qwen candidate. Runs only if the assigned GPU has enough memory.",
    },
    {
        "model_id": "llava-hf/llava-v1.6-34b-hf",
        "trust_remote_code": False,
        "eval_limit_per_class": 2,
        "min_gpu_gb": 38,
        "notes": "Large requested baseline. Usually skipped on free Kaggle unless a very large GPU is assigned.",
    },
]

HIERARCHICAL_MODEL_CANDIDATES = [
    {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 10,
        "min_gpu_gb": 10,
        "notes": "Batch 3 primary hierarchical diagnostic on the best free-tier Qwen model.",
    },
    {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "trust_remote_code": False,
        "eval_limit_per_class": 5,
        "min_gpu_gb": 24,
        "notes": "Batch 3 stronger hierarchical diagnostic if Kaggle assigns two T4 GPUs again.",
    },
]

RUN_HIERARCHICAL_DIAGNOSTIC = True
RUN_FLAT_BENCHMARK = False
RUN_LORA = False
LORA_BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_TRAIN_PER_CLASS = 32
LORA_VAL_PER_CLASS = 8
LORA_EVAL_PER_CLASS = 12
LORA_MAX_STEPS = 64
LORA_GRAD_ACCUM = 4


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def print_banner(message: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{utc_now()} | {message}\n{line}", flush=True)


def record(stage: str, status: str, **details: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": utc_now(),
        "stage": stage,
        "status": status,
        "details": details,
    }
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    print(f"[{payload['created_utc']}] {stage}: {status} {details}", flush=True)


def run_command(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(args)
    record("command", "started", cmd=printable, cwd=str(cwd or Path.cwd()))
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, check=True)
    record("command", "finished", cmd=printable)


def install_dependencies() -> None:
    print_banner("Installing Python dependencies")
    packages = [
        "transformers>=4.52.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.12.0",
        "qwen-vl-utils>=0.0.8",
        "datasets>=2.20.0",
        "pillow>=10,<12",
        "pandas>=2.2,<3",
        "scikit-learn",
        "sentencepiece",
        "protobuf>=4.25,<6",
        "einops",
        "num2words",
    ]
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", *packages])


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "total_memory_gb": 0.0,
        "nvidia_smi": "",
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
            capability = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{capability[0]}.{capability[1]}"
            info["compute_capability_major"] = capability[0]
            info["torch_cuda_supported"] = capability[0] >= 7
            info["device_count"] = torch.cuda.device_count()
            devices = []
            aggregate_memory = 0.0
            for index in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(index)
                device_capability = torch.cuda.get_device_capability(index)
                memory_gb = round(device_props.total_memory / (1024**3), 2)
                aggregate_memory += memory_gb
                devices.append(
                    {
                        "index": index,
                        "name": device_props.name,
                        "memory_gb": memory_gb,
                        "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
                    }
                )
            info["devices"] = devices
            info["aggregate_memory_gb"] = round(aggregate_memory, 2)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not stop the job.
        info["torch_error"] = repr(exc)

    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            text=True,
            capture_output=True,
            check=False,
        )
        info["nvidia_smi"] = completed.stdout[-4000:]
    except Exception as exc:  # noqa: BLE001
        info["nvidia_smi_error"] = repr(exc)

    return info


def clone_repo() -> None:
    print_banner("Cloning Tumor-Database")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "main",
            REPO_URL,
            str(REPO_DIR),
        ],
        env=env,
    )
    record("repo", "cloned", path=str(REPO_DIR))


def create_manifest() -> Path:
    print_banner("Creating strict split manifest")
    run_command([sys.executable, "-m", "src.experiment_pipeline", "create-manifest"], cwd=REPO_DIR)
    run_command([sys.executable, "-m", "src.experiment_pipeline", "summary"], cwd=REPO_DIR)
    manifest = REPO_DIR / "training_logs" / "splits" / "strict_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest was not created: {manifest}")
    shutil.copy2(manifest, OUTPUT_DIR / "strict_manifest.csv")
    summary = summarize_manifest(manifest)
    (OUTPUT_DIR / "strict_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    record("manifest", "created", path=str(manifest), summary=summary)
    return manifest


def summarize_manifest(manifest_path: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['split']}::{row['eight_class']}"
            counts[key] = {"count": counts.get(key, {"count": 0})["count"] + 1}
    return {"rows": sum(item["count"] for item in counts.values()), "counts": counts}


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def balanced_rows(rows: list[dict[str, str]], split: str, per_class: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in LABELS:
        group = [row for row in rows if row["split"] == split and row["eight_class"] == label]
        group = sorted(group, key=lambda row: row["path"])
        selected.extend(group[:per_class])
    return selected


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_image(image_path: Path):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((512, 512))
    return image


def parse_model_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                text = parse_model_text(item.get("content", ""))
                if text:
                    return text
        parts = [parse_model_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url", "video"} or "image" in value:
            return ""
        for key in ("generated_text", "content", "text"):
            if key in value:
                text = parse_model_text(value[key])
                if text:
                    return text
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)


def parse_prediction(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "rationale": text[:300], "strict_json": False}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "rationale": text[:300], "strict_json": False}
    label = str(data.get("label", "INVALID_LABEL"))
    if label not in LABELS:
        label = "INVALID_LABEL"
    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": label,
        "confidence": confidence,
        "rationale": str(data.get("rationale", ""))[:500],
        "strict_json": True,
    }


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None, False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, False
    return data, True


def parse_confidence(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_domain_prediction(text: str) -> dict[str, Any]:
    data, strict_json = parse_json_object(text)
    if not data:
        return {
            "domain": "PARSE_ERROR",
            "confidence": 0.0,
            "rationale": text[:300],
            "strict_json": False,
        }
    domain = str(data.get("domain", data.get("label", "INVALID_DOMAIN"))).strip().lower()
    if domain.startswith("tumor"):
        domain = "tumor"
    elif domain.startswith("dementia") or domain.startswith("alz"):
        domain = "dementia"
    else:
        domain = "INVALID_DOMAIN"
    return {
        "domain": domain,
        "confidence": parse_confidence(data.get("confidence")),
        "rationale": str(data.get("rationale", ""))[:500],
        "strict_json": strict_json,
    }


def normalize_subtype(raw_label: Any, domain: str) -> str:
    label = str(raw_label or "").strip()
    if domain == "tumor":
        label = label.removeprefix("tumor_")
        label_map = {item.lower(): item for item in TUMOR_SUBTYPES}
        return label_map.get(label.lower(), "INVALID_SUBTYPE")
    if domain == "dementia":
        label = label.removeprefix("dementia_")
        label_map = {item.lower(): item for item in DEMENTIA_SUBTYPES}
        return label_map.get(label.lower(), "INVALID_SUBTYPE")
    return "INVALID_SUBTYPE"


def subtype_to_eight_class(subtype: str, domain: str) -> str:
    if domain == "tumor" and subtype in TUMOR_SUBTYPES:
        return f"tumor_{subtype}"
    if domain == "dementia" and subtype in DEMENTIA_SUBTYPES:
        return f"dementia_{subtype}"
    return "INVALID_OR_PARSE"


def parse_subtype_prediction(text: str, domain: str) -> dict[str, Any]:
    data, strict_json = parse_json_object(text)
    if not data:
        return {
            "subtype": "PARSE_ERROR",
            "eight_class": "INVALID_OR_PARSE",
            "confidence": 0.0,
            "rationale": text[:300],
            "strict_json": False,
        }
    subtype = normalize_subtype(data.get("subtype", data.get("label", "INVALID_SUBTYPE")), domain)
    return {
        "subtype": subtype,
        "eight_class": subtype_to_eight_class(subtype, domain),
        "confidence": parse_confidence(data.get("confidence")),
        "rationale": str(data.get("rationale", ""))[:500],
        "strict_json": strict_json,
    }


def make_messages(image: Any) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


def make_domain_messages(image: Any) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DOMAIN_PROMPT},
            ],
        },
    ]


def make_subtype_messages(image: Any, domain: str) -> list[dict[str, Any]]:
    prompt = TUMOR_SUBTYPE_PROMPT if domain == "tumor" else DEMENTIA_SUBTYPE_PROMPT
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def load_vlm_pipeline(candidate: dict[str, Any]):
    import torch
    from transformers import BitsAndBytesConfig, pipeline

    model_id = candidate["model_id"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": quantization_config,
    }
    if "Phi-3.5-vision" in model_id:
        model_kwargs["attn_implementation"] = "eager"

    return pipeline(
        "image-text-to-text",
        model=model_id,
        trust_remote_code=bool(candidate.get("trust_remote_code", False)),
        model_kwargs=model_kwargs,
    )


def evaluate_candidate(candidate: dict[str, Any], all_rows: list[dict[str, str]], gpu: dict[str, Any]) -> dict[str, Any]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model_id = candidate["model_id"]
    safe_name = model_id.replace("/", "__")
    needed_gb = float(candidate.get("min_gpu_gb", 0))
    available_gb = float(gpu.get("aggregate_memory_gb") or gpu.get("total_memory_gb", 0.0))
    if available_gb and available_gb < needed_gb:
        summary = {
            "model_id": model_id,
            "status": "skipped",
            "reason": f"GPU memory {available_gb} GB is below requested {needed_gb} GB",
            "notes": candidate.get("notes", ""),
        }
        (OUTPUT_DIR / f"{safe_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        record("model_eval", "skipped", model_id=model_id, reason=summary["reason"])
        return summary

    print_banner(f"Evaluating {model_id}")
    rows = balanced_rows(all_rows, "test", int(candidate.get("eval_limit_per_class", 4)))
    if not rows:
        raise RuntimeError("No strict test rows available for multimodal evaluation")

    pipe = None
    result_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        pipe = load_vlm_pipeline(candidate)
        for idx, row in enumerate(rows, start=1):
            image_path = REPO_DIR / row["path"]
            image = load_image(image_path)
            messages = make_messages(image)
            raw = pipe(text=messages, max_new_tokens=96)
            generated = raw[0] if isinstance(raw, list) and raw else raw
            text = parse_model_text(generated)
            parsed = parse_prediction(text)
            pred = parsed["label"]
            correct = pred == row["eight_class"]
            result = {
                "model_id": model_id,
                "idx": idx,
                "path": row["path"],
                "true_label": row["eight_class"],
                "pred_label": pred,
                "confidence": parsed["confidence"],
                "correct": correct,
                "rationale": parsed["rationale"],
                "raw_text": text[:1500],
            }
            result_rows.append(result)
            record(
                "model_eval",
                "progress",
                model_id=model_id,
                completed=idx,
                total=len(rows),
                running_accuracy=round(sum(1 for item in result_rows if item["correct"]) / len(result_rows), 4),
            )
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_true = [row["true_label"] for row in result_rows]
    y_pred = [row["pred_label"] for row in result_rows]
    valid_preds = [pred if pred in LABELS else "INVALID_OR_PARSE" for pred in y_pred]
    labels_for_matrix = LABELS + ["INVALID_OR_PARSE"]
    matrix = confusion_matrix(y_true, valid_preds, labels=labels_for_matrix)
    summary = {
        "model_id": model_id,
        "status": "completed",
        "notes": candidate.get("notes", ""),
        "n": len(result_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)) if result_rows else 0.0,
        "strict_json_rate": float(sum(pred in LABELS for pred in y_pred) / len(y_pred)) if y_pred else 0.0,
        "seconds": round(time.time() - started, 2),
        "classification_report": classification_report(y_true, valid_preds, labels=labels_for_matrix, zero_division=0),
        "confusion_matrix_labels": labels_for_matrix,
        "confusion_matrix": matrix.tolist(),
    }
    write_rows_csv(OUTPUT_DIR / f"{safe_name}_eval.csv", result_rows)
    (OUTPUT_DIR / f"{safe_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    record("model_eval", "completed", model_id=model_id, accuracy=summary["accuracy"], n=summary["n"])
    return summary


def value_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def evaluate_hierarchical_candidate(
    candidate: dict[str, Any],
    all_rows: list[dict[str, str]],
    gpu: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model_id = candidate["model_id"]
    safe_name = model_id.replace("/", "__")
    needed_gb = float(candidate.get("min_gpu_gb", 0))
    available_gb = float(gpu.get("aggregate_memory_gb") or gpu.get("total_memory_gb", 0.0))
    if available_gb and available_gb < needed_gb:
        summary = {
            "model_id": model_id,
            "status": "skipped",
            "reason": f"GPU memory {available_gb} GB is below requested {needed_gb} GB",
            "notes": candidate.get("notes", ""),
            "mode": "hierarchical",
        }
        (OUTPUT_DIR / f"{safe_name}_hierarchical_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        record("hierarchical_eval", "skipped", model_id=model_id, reason=summary["reason"])
        return summary

    print_banner(f"Hierarchical evaluation for {model_id}")
    rows = balanced_rows(all_rows, "test", int(candidate.get("eval_limit_per_class", 4)))
    if not rows:
        raise RuntimeError("No strict test rows available for hierarchical multimodal evaluation")

    pipe = None
    result_rows: list[dict[str, Any]] = []
    started = time.time()

    def run_generation(messages: list[dict[str, Any]], max_new_tokens: int = 96) -> str:
        raw = pipe(text=messages, max_new_tokens=max_new_tokens)
        generated = raw[0] if isinstance(raw, list) and raw else raw
        return parse_model_text(generated)

    try:
        pipe = load_vlm_pipeline(candidate)
        for idx, row in enumerate(rows, start=1):
            image_path = REPO_DIR / row["path"]
            image = load_image(image_path)
            true_domain = row.get("domain") or ("tumor" if row["eight_class"].startswith("tumor_") else "dementia")

            domain_text = run_generation(make_domain_messages(image), max_new_tokens=80)
            domain_parsed = parse_domain_prediction(domain_text)
            pred_domain = domain_parsed["domain"]

            routed_subtype_text = ""
            routed_subtype = "INVALID_SUBTYPE"
            pred_eight_class = "INVALID_OR_PARSE"
            subtype_confidence = 0.0
            subtype_rationale = ""
            subtype_strict_json = False
            if pred_domain in DOMAIN_LABELS:
                routed_subtype_text = run_generation(make_subtype_messages(image, pred_domain), max_new_tokens=96)
                routed_subtype_parsed = parse_subtype_prediction(routed_subtype_text, pred_domain)
                routed_subtype = routed_subtype_parsed["subtype"]
                pred_eight_class = routed_subtype_parsed["eight_class"]
                subtype_confidence = routed_subtype_parsed["confidence"]
                subtype_rationale = routed_subtype_parsed["rationale"]
                subtype_strict_json = bool(routed_subtype_parsed["strict_json"])

            if pred_domain == true_domain and pred_eight_class in LABELS:
                oracle_subtype_text = routed_subtype_text
                oracle_subtype = routed_subtype
                oracle_eight_class = pred_eight_class
                oracle_subtype_confidence = subtype_confidence
                oracle_subtype_strict_json = subtype_strict_json
            else:
                oracle_subtype_text = run_generation(make_subtype_messages(image, true_domain), max_new_tokens=96)
                oracle_subtype_parsed = parse_subtype_prediction(oracle_subtype_text, true_domain)
                oracle_subtype = oracle_subtype_parsed["subtype"]
                oracle_eight_class = oracle_subtype_parsed["eight_class"]
                oracle_subtype_confidence = oracle_subtype_parsed["confidence"]
                oracle_subtype_strict_json = bool(oracle_subtype_parsed["strict_json"])

            result = {
                "model_id": model_id,
                "idx": idx,
                "path": row["path"],
                "true_domain": true_domain,
                "true_label": row["eight_class"],
                "pred_domain": pred_domain,
                "pred_label": pred_eight_class,
                "oracle_domain_pred_label": oracle_eight_class,
                "domain_correct": pred_domain == true_domain,
                "hierarchical_correct": pred_eight_class == row["eight_class"],
                "oracle_domain_correct": oracle_eight_class == row["eight_class"],
                "domain_confidence": domain_parsed["confidence"],
                "subtype_confidence": subtype_confidence,
                "oracle_subtype_confidence": oracle_subtype_confidence,
                "routed_subtype": routed_subtype,
                "oracle_subtype": oracle_subtype,
                "domain_strict_json": bool(domain_parsed["strict_json"]),
                "subtype_strict_json": subtype_strict_json,
                "oracle_subtype_strict_json": oracle_subtype_strict_json,
                "domain_rationale": domain_parsed["rationale"],
                "subtype_rationale": subtype_rationale,
                "domain_raw_text": domain_text[:1500],
                "subtype_raw_text": routed_subtype_text[:1500],
                "oracle_subtype_raw_text": oracle_subtype_text[:1500],
            }
            result_rows.append(result)
            record(
                "hierarchical_eval",
                "progress",
                model_id=model_id,
                completed=idx,
                total=len(rows),
                domain_accuracy=round(sum(1 for item in result_rows if item["domain_correct"]) / len(result_rows), 4),
                hierarchical_accuracy=round(
                    sum(1 for item in result_rows if item["hierarchical_correct"]) / len(result_rows),
                    4,
                ),
                oracle_domain_accuracy=round(
                    sum(1 for item in result_rows if item["oracle_domain_correct"]) / len(result_rows),
                    4,
                ),
            )
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_true_domain = [row["true_domain"] for row in result_rows]
    y_pred_domain = [
        row["pred_domain"] if row["pred_domain"] in DOMAIN_LABELS else "INVALID_OR_PARSE"
        for row in result_rows
    ]
    domain_labels_for_matrix = DOMAIN_LABELS + ["INVALID_OR_PARSE"]
    y_true_eight = [row["true_label"] for row in result_rows]
    y_pred_eight = [
        row["pred_label"] if row["pred_label"] in LABELS else "INVALID_OR_PARSE"
        for row in result_rows
    ]
    y_oracle_eight = [
        row["oracle_domain_pred_label"] if row["oracle_domain_pred_label"] in LABELS else "INVALID_OR_PARSE"
        for row in result_rows
    ]
    eight_labels_for_matrix = LABELS + ["INVALID_OR_PARSE"]

    domain_matrix = confusion_matrix(y_true_domain, y_pred_domain, labels=domain_labels_for_matrix)
    hierarchical_matrix = confusion_matrix(y_true_eight, y_pred_eight, labels=eight_labels_for_matrix)
    oracle_matrix = confusion_matrix(y_true_eight, y_oracle_eight, labels=eight_labels_for_matrix)
    summary = {
        "model_id": model_id,
        "status": "completed",
        "mode": "hierarchical",
        "notes": candidate.get("notes", ""),
        "n": len(result_rows),
        "domain_accuracy": float(accuracy_score(y_true_domain, y_pred_domain)) if result_rows else 0.0,
        "accuracy": float(accuracy_score(y_true_eight, y_pred_eight)) if result_rows else 0.0,
        "hierarchical_accuracy": float(accuracy_score(y_true_eight, y_pred_eight)) if result_rows else 0.0,
        "oracle_domain_accuracy": float(accuracy_score(y_true_eight, y_oracle_eight)) if result_rows else 0.0,
        "domain_strict_json_rate": (
            float(sum(bool(row["domain_strict_json"]) for row in result_rows) / len(result_rows))
            if result_rows
            else 0.0
        ),
        "subtype_strict_json_rate": (
            float(sum(bool(row["subtype_strict_json"]) for row in result_rows) / len(result_rows))
            if result_rows
            else 0.0
        ),
        "oracle_subtype_strict_json_rate": (
            float(sum(bool(row["oracle_subtype_strict_json"]) for row in result_rows) / len(result_rows))
            if result_rows
            else 0.0
        ),
        "seconds": round(time.time() - started, 2),
        "domain_pred_counts": value_counts(y_pred_domain),
        "pred_counts": value_counts(y_pred_eight),
        "oracle_pred_counts": value_counts(y_oracle_eight),
        "domain_classification_report": classification_report(
            y_true_domain,
            y_pred_domain,
            labels=domain_labels_for_matrix,
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_true_eight,
            y_pred_eight,
            labels=eight_labels_for_matrix,
            zero_division=0,
        ),
        "oracle_domain_classification_report": classification_report(
            y_true_eight,
            y_oracle_eight,
            labels=eight_labels_for_matrix,
            zero_division=0,
        ),
        "domain_confusion_matrix_labels": domain_labels_for_matrix,
        "domain_confusion_matrix": domain_matrix.tolist(),
        "confusion_matrix_labels": eight_labels_for_matrix,
        "confusion_matrix": hierarchical_matrix.tolist(),
        "oracle_domain_confusion_matrix": oracle_matrix.tolist(),
    }
    write_rows_csv(OUTPUT_DIR / f"{safe_name}_hierarchical_eval.csv", result_rows)
    (OUTPUT_DIR / f"{safe_name}_hierarchical_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    record(
        "hierarchical_eval",
        "completed",
        model_id=model_id,
        domain_accuracy=summary["domain_accuracy"],
        hierarchical_accuracy=summary["hierarchical_accuracy"],
        oracle_domain_accuracy=summary["oracle_domain_accuracy"],
        n=summary["n"],
    )
    return summary


def build_lora_jsonl(all_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    print_banner("Building LoRA JSONL data")
    data_dir = OUTPUT_DIR / "lora_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = balanced_rows(all_rows, "train", LORA_TRAIN_PER_CLASS)
    val_rows = balanced_rows(all_rows, "val", LORA_VAL_PER_CLASS)

    def convert(row: dict[str, str]) -> dict[str, Any]:
        return {
            "image": row["path"],
            "label": row["eight_class"],
            "answer_json": {
                "label": row["eight_class"],
                "confidence": 1.0,
                "rationale": "MRI appearance matches the labeled research class.",
            },
        }

    train_path = data_dir / "multimodal_lora_train.jsonl"
    val_path = data_dir / "multimodal_lora_val.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(convert(row), ensure_ascii=True) + "\n")
    with val_path.open("w", encoding="utf-8") as handle:
        for row in val_rows:
            handle.write(json.dumps(convert(row), ensure_ascii=True) + "\n")
    record("lora_data", "created", train_rows=len(train_rows), val_rows=len(val_rows))
    return train_path, val_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qwen_messages_for_training(example: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    image_path = str(REPO_DIR / example["image"])
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(example["answer_json"], ensure_ascii=False)}],
            }
        )
    return messages


def train_qwen_lora(train_jsonl: Path, val_jsonl: Path, gpu: dict[str, Any]) -> dict[str, Any]:
    if not RUN_LORA:
        return {"status": "skipped", "reason": "RUN_LORA is false"}
    if not gpu.get("cuda_available"):
        return {"status": "skipped", "reason": "CUDA is not available"}
    if float(gpu.get("total_memory_gb", 0.0)) < 10:
        return {"status": "skipped", "reason": "GPU memory is too small for Qwen LoRA"}

    print_banner(f"Training LoRA adapter for {LORA_BASE_MODEL_ID}")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    adapter_dir = OUTPUT_DIR / "lora_adapter_qwen25vl_3b_mri"
    train_examples = read_jsonl(train_jsonl)
    val_examples = read_jsonl(val_jsonl)
    random.Random(42).shuffle(train_examples)
    train_examples = train_examples[: max(1, min(len(train_examples), LORA_MAX_STEPS * LORA_GRAD_ACCUM))]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        LORA_BASE_MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        LORA_BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
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
    model.train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=2e-4)
    device = next(model.parameters()).device
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    def make_batch(example: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompt_messages = qwen_messages_for_training(example, include_answer=False)
        full_messages = qwen_messages_for_training(example, include_answer=True)
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        full_inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = full_inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in full_inputs.items()}

    completed_updates = 0
    for step, example in enumerate(train_examples, start=1):
        batch = make_batch(example)
        outputs = model(**batch)
        loss = outputs.loss / LORA_GRAD_ACCUM
        loss.backward()
        losses.append(float(outputs.loss.detach().cpu()))
        if step % LORA_GRAD_ACCUM == 0 or step == len(train_examples):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_updates += 1
            record(
                "lora_train",
                "progress",
                update=completed_updates,
                max_updates=LORA_MAX_STEPS,
                examples_seen=step,
                loss=round(losses[-1], 6),
            )
        if completed_updates >= LORA_MAX_STEPS:
            break

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    loss_summary = {
        "status": "completed",
        "base_model_id": LORA_BASE_MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "optimizer_updates": completed_updates,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }

    (adapter_dir / "training_summary.json").write_text(json.dumps(loss_summary, indent=2), encoding="utf-8")
    record("lora_train", "completed", final_loss=loss_summary["final_loss"], adapter_dir=str(adapter_dir))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss_summary


def evaluate_qwen_lora_adapter(adapter_dir: Path, all_rows: list[dict[str, str]], gpu: dict[str, Any]) -> dict[str, Any]:
    if not adapter_dir.exists():
        return {"status": "skipped", "reason": f"Adapter directory does not exist: {adapter_dir}"}
    if not gpu.get("cuda_available"):
        return {"status": "skipped", "reason": "CUDA is not available"}

    print_banner(f"Evaluating LoRA adapter {adapter_dir.name}")
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    rows = balanced_rows(all_rows, "test", LORA_EVAL_PER_CLASS)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(adapter_dir),
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        LORA_BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    device = next(model.parameters()).device

    result_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for idx, row in enumerate(rows, start=1):
            example = {"image": row["path"], "answer_json": {"label": row["eight_class"]}}
            messages = qwen_messages_for_training(example, include_answer=False)
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=96)
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
            ]
            text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed = parse_prediction(text)
            pred = parsed["label"]
            result = {
                "model_id": f"{LORA_BASE_MODEL_ID}+LoRA",
                "idx": idx,
                "path": row["path"],
                "true_label": row["eight_class"],
                "pred_label": pred,
                "confidence": parsed["confidence"],
                "correct": pred == row["eight_class"],
                "rationale": parsed["rationale"],
                "raw_text": text[:1500],
            }
            result_rows.append(result)
            record(
                "lora_eval",
                "progress",
                completed=idx,
                total=len(rows),
                running_accuracy=round(sum(1 for item in result_rows if item["correct"]) / len(result_rows), 4),
            )
    finally:
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_true = [row["true_label"] for row in result_rows]
    y_pred = [row["pred_label"] for row in result_rows]
    valid_preds = [pred if pred in LABELS else "INVALID_OR_PARSE" for pred in y_pred]
    labels_for_matrix = LABELS + ["INVALID_OR_PARSE"]
    matrix = confusion_matrix(y_true, valid_preds, labels=labels_for_matrix)
    summary = {
        "model_id": f"{LORA_BASE_MODEL_ID}+LoRA",
        "status": "completed",
        "adapter_dir": str(adapter_dir),
        "n": len(result_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)) if result_rows else 0.0,
        "strict_json_rate": float(sum(pred in LABELS for pred in y_pred) / len(y_pred)) if y_pred else 0.0,
        "seconds": round(time.time() - started, 2),
        "classification_report": classification_report(y_true, valid_preds, labels=labels_for_matrix, zero_division=0),
        "confusion_matrix_labels": labels_for_matrix,
        "confusion_matrix": matrix.tolist(),
    }
    write_rows_csv(OUTPUT_DIR / "qwen25vl_3b_lora_eval.csv", result_rows)
    (OUTPUT_DIR / "qwen25vl_3b_lora_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    record("lora_eval", "completed", accuracy=summary["accuracy"], n=summary["n"])
    return summary


def package_outputs() -> None:
    print_banner("Packaging outputs")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", OUTPUT_DIR)
    record("package", "created", zip_path=str(ZIP_PATH))


def clean_large_working_dirs() -> None:
    for path in (REPO_DIR, OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final: dict[str, Any] = {
        "created_utc": utc_now(),
        "repo_url": REPO_URL,
        "status": "started",
        "models": [],
        "hierarchical_models": [],
        "run_config": {
            "run_hierarchical_diagnostic": RUN_HIERARCHICAL_DIAGNOSTIC,
            "run_flat_benchmark": RUN_FLAT_BENCHMARK,
            "run_lora": RUN_LORA,
        },
    }
    try:
        print_banner("Kaggle multimodal tumor/dementia job starting")
        record("job", "started")
        install_dependencies()
        gpu = gpu_info()
        final["gpu"] = gpu
        record("gpu", "detected", **gpu)

        if gpu.get("cuda_available") and not gpu.get("torch_cuda_supported", True):
            final["status"] = "unsupported_gpu"
            final["message"] = (
                "Kaggle assigned a pre-Volta GPU. The current Kaggle PyTorch image "
                "does not support this GPU for the selected VLM workflow. Resubmit "
                "with --accelerator NvidiaTeslaT4 or a newer GPU."
            )
            record("gpu", "unsupported", **gpu)
            return

        if not gpu.get("cuda_available"):
            final["status"] = "no_cuda"
            final["message"] = "Kaggle did not provide CUDA; multimodal VLM evaluation was skipped."
            record("gpu", "no_cuda")
            return

        clone_repo()
        manifest_path = create_manifest()
        rows = read_manifest_rows(manifest_path)
        eval_sample = balanced_rows(rows, "test", 10)
        write_rows_csv(OUTPUT_DIR / "multimodal_eval_sample.csv", eval_sample)

        if RUN_HIERARCHICAL_DIAGNOSTIC:
            for candidate in HIERARCHICAL_MODEL_CANDIDATES:
                try:
                    summary = evaluate_hierarchical_candidate(candidate, rows, gpu)
                except Exception as exc:  # noqa: BLE001
                    summary = {
                        "model_id": candidate["model_id"],
                        "status": "failed",
                        "mode": "hierarchical",
                        "error": repr(exc),
                        "traceback": traceback.format_exc()[-8000:],
                        "notes": candidate.get("notes", ""),
                    }
                    safe_name = candidate["model_id"].replace("/", "__")
                    (OUTPUT_DIR / f"{safe_name}_hierarchical_summary.json").write_text(
                        json.dumps(summary, indent=2),
                        encoding="utf-8",
                    )
                    record("hierarchical_eval", "failed", model_id=candidate["model_id"], error=repr(exc))
                final["hierarchical_models"].append(summary)
        else:
            record("hierarchical_eval", "skipped", reason="RUN_HIERARCHICAL_DIAGNOSTIC is false")

        if RUN_FLAT_BENCHMARK:
            for candidate in MODEL_CANDIDATES:
                try:
                    summary = evaluate_candidate(candidate, rows, gpu)
                except Exception as exc:  # noqa: BLE001
                    summary = {
                        "model_id": candidate["model_id"],
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc()[-8000:],
                        "notes": candidate.get("notes", ""),
                    }
                    safe_name = candidate["model_id"].replace("/", "__")
                    (OUTPUT_DIR / f"{safe_name}_summary.json").write_text(
                        json.dumps(summary, indent=2),
                        encoding="utf-8",
                    )
                    record("model_eval", "failed", model_id=candidate["model_id"], error=repr(exc))
                final["models"].append(summary)
        else:
            record("model_eval", "skipped", reason="RUN_FLAT_BENCHMARK is false for this batch")

        if RUN_LORA:
            train_jsonl, val_jsonl = build_lora_jsonl(rows)
            try:
                final["lora"] = train_qwen_lora(train_jsonl, val_jsonl, final.get("gpu", {}))
            except Exception as exc:  # noqa: BLE001
                final["lora"] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc()[-8000:],
                }
                record("lora_train", "failed", error=repr(exc))

            if final.get("lora", {}).get("status") == "completed":
                try:
                    adapter_path = Path(str(final["lora"]["adapter_dir"]))
                    final["lora_eval"] = evaluate_qwen_lora_adapter(adapter_path, rows, final.get("gpu", {}))
                except Exception as exc:  # noqa: BLE001
                    final["lora_eval"] = {
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc()[-8000:],
                    }
                    record("lora_eval", "failed", error=repr(exc))
        else:
            final["lora"] = {"status": "skipped", "reason": "RUN_LORA is false for this batch"}
            record("lora_train", "skipped", reason=final["lora"]["reason"])

        completed_models = [item for item in final["models"] if item.get("status") == "completed"]
        if completed_models:
            final["best_zero_shot_model"] = sorted(
                completed_models,
                key=lambda item: (float(item.get("accuracy", 0.0)), float(item.get("strict_json_rate", 0.0))),
                reverse=True,
            )[0]
        completed_hierarchical = [
            item for item in final["hierarchical_models"] if item.get("status") == "completed"
        ]
        if completed_hierarchical:
            final["best_hierarchical_model"] = sorted(
                completed_hierarchical,
                key=lambda item: (
                    float(item.get("hierarchical_accuracy", item.get("accuracy", 0.0))),
                    float(item.get("domain_accuracy", 0.0)),
                    float(item.get("oracle_domain_accuracy", 0.0)),
                ),
                reverse=True,
            )[0]
        final["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        final["status"] = "failed"
        final["error"] = repr(exc)
        final["traceback"] = traceback.format_exc()[-12000:]
        record("job", "failed", error=repr(exc))
    finally:
        SUMMARY_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
        package_outputs()
        clean_large_working_dirs()
        print_banner("Kaggle job finished")
        print(json.dumps(final, indent=2), flush=True)
        print(f"\nMain zip output: {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
