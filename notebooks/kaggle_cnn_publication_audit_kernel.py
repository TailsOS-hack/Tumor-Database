"""Kaggle CNN publication audit and regularized retraining job.

This script is intended for a Kaggle script kernel. It clones the repo, rebuilds
the strict manifest, runs leakage/overfitting audit checks, evaluates available
current checkpoints across train/val/test, then launches a regularized CNN
training suite for a publication robustness comparison.
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
OUTPUT_DIR = WORK_DIR / "tumor_cnn_dedup_retrain_outputs"
PROGRESS_PATH = OUTPUT_DIR / "progress.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "cnn_publication_audit_summary.json"
ZIP_PATH = WORK_DIR / "tumor_cnn_dedup_retrain_outputs.zip"

CURRENT_EVAL_DIR = REPO_DIR / "training_logs" / "experiments_dedup_current_eval"
DEFAULT_EXPERIMENTS_DIR = REPO_DIR / "training_logs" / "experiments"
REGULARIZED_EXPERIMENT_DIR = REPO_DIR / "training_logs" / "experiments_dedup_regularized"
PRETRAIN_AUDIT_DIR = REPO_DIR / "training_logs" / "publication_audit" / "dedup_pre_regularized"
REGULARIZED_AUDIT_DIR = REPO_DIR / "training_logs" / "publication_audit" / "dedup_regularized"

REGULARIZED_EPOCHS = 20
REGULARIZED_BATCH_SIZE = 32
REGULARIZED_NUM_WORKERS = 2
REGULARIZED_LEARNING_RATE = "5e-5"
REGULARIZED_WEIGHT_DECAY = "0.001"
REGULARIZED_LABEL_SMOOTHING = "0.05"
REGULARIZED_RANDOM_ERASING = "0.10"
REGULARIZED_PATIENCE = "4"
REGULARIZED_MIN_DELTA = "0.0005"

TASKS = ["binary", "tumor", "dementia", "eight_class"]
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
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        record("command", "failed", cmd=printable, returncode=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, args)
    record("command", "finished", cmd=printable, returncode=completed.returncode)
    return completed


def install_dependencies() -> None:
    print_banner("Installing audit dependencies")
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


def pull_current_checkpoints() -> dict[str, Any]:
    print_banner("Pulling current LFS checkpoints when available")
    lfs_check = run_command(["git", "lfs", "version"], cwd=REPO_DIR, check=False)
    if lfs_check.returncode != 0:
        return {"status": "skipped", "reason": "git-lfs is not available in the Kaggle image"}
    run_command(["git", "lfs", "install", "--local"], cwd=REPO_DIR)
    include = ",".join(MODEL_PATHS.values())
    completed = run_command(["git", "lfs", "pull", "--include", include], cwd=REPO_DIR, check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    checkpoint_sizes = {
        name: (REPO_DIR / rel_path).stat().st_size if (REPO_DIR / rel_path).exists() else 0
        for name, rel_path in MODEL_PATHS.items()
    }
    return {
        "status": status,
        "returncode": completed.returncode,
        "checkpoint_sizes": checkpoint_sizes,
    }


def create_manifest() -> None:
    print_banner("Rebuilding strict manifest")
    run_command([sys.executable, "-m", "src.experiment_pipeline", "create-manifest"], cwd=REPO_DIR)
    run_command([sys.executable, "-m", "src.experiment_pipeline", "summary"], cwd=REPO_DIR)


def run_publication_audit(output_dir: Path, experiments_dir: Path) -> dict[str, Any]:
    print_banner(f"Running publication audit: {output_dir.name}")
    run_command(
        [
            sys.executable,
            "scripts/publication_audit.py",
            "--manifest",
            "training_logs/splits/strict_manifest.csv",
            "--experiments-dir",
            str(experiments_dir.relative_to(REPO_DIR)),
            "--summary-json",
            "docs/colab_publication_summary.json",
            "--output-dir",
            str(output_dir.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    summary_path = output_dir / "audit_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}


def checkpoint_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024 * 1024


def evaluate_current_checkpoints() -> dict[str, Any]:
    print_banner("Evaluating current checkpoints across train/val/test")
    results: dict[str, Any] = {"status": "started", "tasks": {}}
    ready = {task: checkpoint_ready(REPO_DIR / model_path) for task, model_path in MODEL_PATHS.items()}
    results["checkpoint_ready"] = ready
    if not all(ready.values()):
        results["status"] = "skipped"
        results["reason"] = "One or more current checkpoint files were unavailable or still LFS pointers."
        record("current_eval", "skipped", ready=ready)
        return results

    for task in TASKS:
        results["tasks"][task] = {}
        for split in ["train", "val", "test"]:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.experiment_pipeline",
                    "test",
                    "--task",
                    task,
                    "--split",
                    split,
                    "--batch-size",
                    "64",
                    "--num-workers",
                    "2",
                    "--output-dir",
                    str(CURRENT_EVAL_DIR.relative_to(REPO_DIR)),
                ],
                cwd=REPO_DIR,
            )
            metrics_path = CURRENT_EVAL_DIR / task / f"{split}_evaluation" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results["tasks"][task][split] = {
                "accuracy": metrics.get("accuracy"),
                "loss": metrics.get("loss"),
                "n": metrics.get("n"),
            }

    for split in ["train", "val", "test"]:
        run_command(
            [
                sys.executable,
                "-m",
                "src.experiment_pipeline",
                "evaluate-hierarchical",
                "--split",
                split,
                "--batch-size",
                "64",
                "--num-workers",
                "2",
                "--output-dir",
                str(CURRENT_EVAL_DIR.relative_to(REPO_DIR)),
            ],
            cwd=REPO_DIR,
        )
    results["status"] = "completed"
    return results


def train_regularized_suite() -> dict[str, Any]:
    print_banner("Training regularized CNN suite")
    run_command(
        [
            sys.executable,
            "-m",
            "src.experiment_pipeline",
            "suite",
            "--epochs",
            str(REGULARIZED_EPOCHS),
            "--batch-size",
            str(REGULARIZED_BATCH_SIZE),
            "--num-workers",
            str(REGULARIZED_NUM_WORKERS),
            "--learning-rate",
            REGULARIZED_LEARNING_RATE,
            "--weight-decay",
            REGULARIZED_WEIGHT_DECAY,
            "--label-smoothing",
            REGULARIZED_LABEL_SMOOTHING,
            "--random-erasing",
            REGULARIZED_RANDOM_ERASING,
            "--early-stop-patience",
            REGULARIZED_PATIENCE,
            "--early-stop-min-delta",
            REGULARIZED_MIN_DELTA,
            "--output-dir",
            str(REGULARIZED_EXPERIMENT_DIR.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    run_command(
        [
            sys.executable,
            "scripts/collect_publication_results.py",
            "--output-dir",
            str(REGULARIZED_EXPERIMENT_DIR.relative_to(REPO_DIR)),
        ],
        cwd=REPO_DIR,
    )
    return {
        "status": "completed",
        "epochs_requested": REGULARIZED_EPOCHS,
        "batch_size": REGULARIZED_BATCH_SIZE,
        "learning_rate": REGULARIZED_LEARNING_RATE,
        "weight_decay": REGULARIZED_WEIGHT_DECAY,
        "label_smoothing": REGULARIZED_LABEL_SMOOTHING,
        "random_erasing": REGULARIZED_RANDOM_ERASING,
        "early_stop_patience": REGULARIZED_PATIENCE,
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
    print_banner("Packaging outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_if_exists(REPO_DIR / "training_logs" / "splits", OUTPUT_DIR / "training_logs" / "splits")
    copy_if_exists(CURRENT_EVAL_DIR, OUTPUT_DIR / "training_logs" / "experiments_current_eval")
    copy_if_exists(REGULARIZED_EXPERIMENT_DIR, OUTPUT_DIR / "training_logs" / "experiments_regularized")
    copy_if_exists(REPO_DIR / "training_logs" / "publication_audit", OUTPUT_DIR / "training_logs" / "publication_audit")
    copy_if_exists(REPO_DIR / "docs" / "colab_publication_summary.json", OUTPUT_DIR / "docs" / "colab_publication_summary.json")
    for model_path in MODEL_PATHS.values():
        copy_if_exists(REPO_DIR / model_path, OUTPUT_DIR / model_path)
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
        "regularized_config": {
            "epochs": REGULARIZED_EPOCHS,
            "batch_size": REGULARIZED_BATCH_SIZE,
            "num_workers": REGULARIZED_NUM_WORKERS,
            "learning_rate": REGULARIZED_LEARNING_RATE,
            "weight_decay": REGULARIZED_WEIGHT_DECAY,
            "label_smoothing": REGULARIZED_LABEL_SMOOTHING,
            "random_erasing": REGULARIZED_RANDOM_ERASING,
            "patience": REGULARIZED_PATIENCE,
        },
    }
    try:
        print_banner("Kaggle CNN publication audit job starting")
        record("job", "started")
        install_dependencies()
        final["gpu"] = gpu_info()
        record("gpu", "detected", **final["gpu"])
        clone_repo()
        final["lfs"] = pull_current_checkpoints()
        create_manifest()
        final["pre_regularized_audit"] = run_publication_audit(PRETRAIN_AUDIT_DIR, DEFAULT_EXPERIMENTS_DIR)
        final["current_eval"] = evaluate_current_checkpoints()
        final["regularized_training"] = train_regularized_suite()
        final["regularized_audit"] = run_publication_audit(REGULARIZED_AUDIT_DIR, REGULARIZED_EXPERIMENT_DIR)
        final["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        final["status"] = "failed"
        final["error"] = repr(exc)
        final["traceback"] = traceback.format_exc()[-12000:]
        record("job", "failed", error=repr(exc))
    finally:
        package_outputs(final)
        clean_large_working_dirs()
        print_banner("Kaggle CNN publication audit job finished")
        print(json.dumps(final, indent=2), flush=True)
        print(f"\nMain zip output: {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
