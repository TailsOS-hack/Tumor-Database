"""Kaggle CNN perceptual-hash sensitivity retraining job.

This script is intended for a Kaggle script kernel. It clones the repo, creates
an alternate strict manifest that groups identical dHash fingerprints into one
split, retrains the CNN suite on that stricter manifest, audits the resulting
split, and packages the outputs for import.

The goal is not to replace the accepted exact-deduplicated baseline by default.
It is a reviewer-facing sensitivity check for the remaining perceptual-hash
overlap warning.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WORK_DIR = Path("/kaggle/working")
REPO_URL = "https://github.com/TailsOS-hack/Tumor-Database.git"
REPO_DIR = WORK_DIR / "Tumor-Database"
OUTPUT_DIR = WORK_DIR / "tumor_cnn_perceptual_sensitivity_outputs"
PROGRESS_PATH = OUTPUT_DIR / "progress.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "cnn_perceptual_sensitivity_summary.json"
ZIP_PATH = WORK_DIR / "tumor_cnn_perceptual_sensitivity_outputs.zip"

PERCEPTUAL_MANIFEST = REPO_DIR / "training_logs" / "splits" / "perceptual_strict_manifest.csv"
EXPERIMENT_DIR = REPO_DIR / "training_logs" / "experiments_perceptual_regularized"
PRETRAIN_AUDIT_DIR = REPO_DIR / "training_logs" / "publication_audit" / "perceptual_pretrain"
FINAL_AUDIT_DIR = REPO_DIR / "training_logs" / "publication_audit" / "perceptual_regularized"

EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = "5e-5"
WEIGHT_DECAY = "0.001"
LABEL_SMOOTHING = "0.05"
RANDOM_ERASING = "0.10"
PATIENCE = "4"
MIN_DELTA = "0.0005"

MODEL_PATHS = {
    "binary": "models/binary_router.pt",
    "tumor": "models/brain_tumor_classifier.pt",
    "dementia": "models/alzheimers_classifier.pt",
    "eight_class": "models/single_8class_classifier.pt",
}


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


def run_command(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(args)
    record("command", "started", cmd=printable, cwd=str(cwd or Path.cwd()))
    completed = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, check=False)
    if check and completed.returncode != 0:
        record("command", "failed", cmd=printable, returncode=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, args)
    record("command", "finished", cmd=printable, returncode=completed.returncode)
    return completed


def install_dependencies() -> None:
    print_banner("Installing sensitivity-job dependencies")
    packages = [
        "pillow>=10,<12",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pandas",
        "tqdm",
    ]
    run_command([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *packages])


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices = []
        total_memory = 0.0
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gb = round(props.total_memory / (1024**3), 2)
            total_memory += memory_gb
            devices.append({"index": index, "name": props.name, "memory_gb": memory_gb})
        info["devices"] = devices
        info["aggregate_memory_gb"] = round(total_memory, 2)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        completed = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
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
    record("command", "started", cmd=f"git clone {REPO_URL} {REPO_DIR}", cwd=str(WORK_DIR))
    subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", "--branch", "main", REPO_URL, str(REPO_DIR)],
        cwd=str(WORK_DIR),
        env=env,
        check=True,
    )
    record("repo", "cloned", path=str(REPO_DIR))


def create_perceptual_manifest() -> dict[str, Any]:
    print_banner("Creating perceptual-hash grouped strict manifest")
    run_command(
        [
            sys.executable,
            "-m",
            "src.experiment_pipeline",
            "create-manifest",
            "--manifest",
            str(PERCEPTUAL_MANIFEST.relative_to(REPO_DIR)),
            "--dedupe-perceptual-hash",
        ],
        cwd=REPO_DIR,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.experiment_pipeline",
            "summary",
            "--manifest",
            str(PERCEPTUAL_MANIFEST.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    return {"status": "completed", "manifest_path": str(PERCEPTUAL_MANIFEST.relative_to(REPO_DIR))}


def run_publication_audit(output_dir: Path, experiments_dir: Path) -> dict[str, Any]:
    print_banner(f"Running publication audit: {output_dir.name}")
    summary_json = experiments_dir / "publication_summary.json"
    args = [
        sys.executable,
        "scripts/publication_audit.py",
        "--manifest",
        str(PERCEPTUAL_MANIFEST.relative_to(REPO_DIR)),
        "--experiments-dir",
        str(experiments_dir.relative_to(REPO_DIR)),
        "--output-dir",
        str(output_dir.relative_to(REPO_DIR)),
    ]
    if summary_json.exists():
        args.extend(["--summary-json", str(summary_json.relative_to(REPO_DIR))])
    run_command(args, cwd=REPO_DIR)
    audit_summary = output_dir / "audit_summary.json"
    return json.loads(audit_summary.read_text(encoding="utf-8")) if audit_summary.exists() else {}


def train_regularized_suite() -> dict[str, Any]:
    print_banner("Training regularized CNN suite on perceptual manifest")
    run_command(
        [
            sys.executable,
            "-m",
            "src.experiment_pipeline",
            "suite",
            "--manifest",
            str(PERCEPTUAL_MANIFEST.relative_to(REPO_DIR)),
            "--epochs",
            str(EPOCHS),
            "--batch-size",
            str(BATCH_SIZE),
            "--num-workers",
            str(NUM_WORKERS),
            "--learning-rate",
            LEARNING_RATE,
            "--weight-decay",
            WEIGHT_DECAY,
            "--label-smoothing",
            LABEL_SMOOTHING,
            "--random-erasing",
            RANDOM_ERASING,
            "--early-stop-patience",
            PATIENCE,
            "--early-stop-min-delta",
            MIN_DELTA,
            "--output-dir",
            str(EXPERIMENT_DIR.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    run_command(
        [
            sys.executable,
            "scripts/collect_publication_results.py",
            "--output-dir",
            str(EXPERIMENT_DIR.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    return {
        "status": "completed",
        "epochs_requested": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "random_erasing": RANDOM_ERASING,
        "early_stop_patience": PATIENCE,
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def package_outputs(final: dict[str, Any]) -> None:
    print_banner("Packaging sensitivity outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_if_exists(PERCEPTUAL_MANIFEST, OUTPUT_DIR / "training_logs" / "splits" / "perceptual_strict_manifest.csv")
    copy_if_exists(EXPERIMENT_DIR, OUTPUT_DIR / "training_logs" / "experiments_perceptual_regularized")
    copy_if_exists(PRETRAIN_AUDIT_DIR, OUTPUT_DIR / "training_logs" / "publication_audit" / "perceptual_pretrain")
    copy_if_exists(FINAL_AUDIT_DIR, OUTPUT_DIR / "training_logs" / "publication_audit" / "perceptual_regularized")
    for name, rel_path in MODEL_PATHS.items():
        copy_if_exists(REPO_DIR / rel_path, OUTPUT_DIR / "models_perceptual" / f"{name}.pt")
    SUMMARY_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
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
        "sensitivity_question": "Do identical perceptual dHash groups crossing splits materially inflate CNN metrics?",
        "regularized_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "random_erasing": RANDOM_ERASING,
            "patience": PATIENCE,
        },
    }
    try:
        print_banner("Kaggle CNN perceptual sensitivity job starting")
        record("job", "started")
        install_dependencies()
        final["gpu"] = gpu_info()
        record("gpu", "detected", **final["gpu"])
        clone_repo()
        final["manifest"] = create_perceptual_manifest()
        final["pretrain_audit"] = run_publication_audit(PRETRAIN_AUDIT_DIR, EXPERIMENT_DIR)
        final["regularized_training"] = train_regularized_suite()
        final["regularized_audit"] = run_publication_audit(FINAL_AUDIT_DIR, EXPERIMENT_DIR)
        final["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        final["status"] = "failed"
        final["error"] = repr(exc)
        final["traceback"] = traceback.format_exc()[-12000:]
        record("job", "failed", error=repr(exc))
    finally:
        package_outputs(final)
        clean_large_working_dirs()
        print_banner("Kaggle CNN perceptual sensitivity job finished")
        print(json.dumps(final, indent=2), flush=True)
        print(f"\nMain zip output: {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
