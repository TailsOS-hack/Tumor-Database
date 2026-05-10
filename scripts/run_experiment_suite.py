#!/usr/bin/env python3
"""Remote-safe launcher for the full tumor/dementia experiment suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    command = [sys.executable, "-m", "src.experiment_pipeline", *args]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def announce(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("\n" + "-" * 80, flush=True)
    print(f"[{timestamp}] {message}", flush=True)
    print("-" * 80, flush=True)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def write_run_metadata(mode: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": cuda_available(),
        "created_epoch": int(time.time()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_results(output_dir: Path) -> None:
    command = [sys.executable, "scripts/collect_publication_results.py", "--output-dir", str(output_dir)]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_manifest() -> None:
    announce("Creating strict train/val/test manifest")
    run(["create-manifest"])
    announce("Summarizing strict manifest")
    run(["summary"])


def run_smoke(output_dir: Path) -> None:
    run_manifest()
    common = [
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--max-samples-per-class",
        "2",
        "--smoke-test",
        "--no-pretrained",
        "--output-dir",
        str(output_dir),
    ]
    for task in ["binary", "tumor", "dementia"]:
        announce(f"Smoke training {task} model")
        run(["train", "--task", task, *common])

    announce("Smoke evaluating hierarchical pipeline")
    run(
        [
            "evaluate-hierarchical",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--max-samples-per-class",
            "2",
            "--smoke-test",
            "--output-dir",
            str(output_dir),
        ]
    )
    announce("Smoke training single 8-class baseline")
    run(["train", "--task", "eight_class", *common])
    announce("Collecting smoke publication summary")
    collect_results(output_dir)


def run_full(output_dir: Path, epochs: int, batch_size: int, num_workers: int, allow_cpu_full: bool) -> None:
    if not cuda_available() and not allow_cpu_full:
        raise SystemExit(
            "Full training needs a CUDA/GPU runner. Re-run on a GPU/self-hosted runner "
            "or pass --allow-cpu-full if you intentionally want a very slow CPU job."
        )

    run_manifest()
    common = [
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--output-dir",
        str(output_dir),
        "--pretrained",
    ]
    for task in ["binary", "tumor", "dementia"]:
        announce(f"Full training {task} model")
        run(["train", "--task", task, *common])

    announce("Evaluating full hierarchical pipeline on strict test split")
    run(["evaluate-hierarchical", "--num-workers", str(num_workers), "--output-dir", str(output_dir)])
    announce("Full training single 8-class baseline")
    run(["train", "--task", "eight_class", *common])
    announce("Collecting full publication summary")
    collect_results(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["manifest", "smoke", "full"], default=os.environ.get("EXPERIMENT_MODE", "smoke"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EXPERIMENT_EPOCHS", "30")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("EXPERIMENT_BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("EXPERIMENT_NUM_WORKERS", "0")))
    parser.add_argument("--output-dir", default=os.environ.get("EXPERIMENT_OUTPUT_DIR", "training_logs/experiments"))
    parser.add_argument("--allow-cpu-full", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    write_run_metadata(args.mode, output_dir)

    if args.mode == "manifest":
        run_manifest()
    elif args.mode == "smoke":
        run_smoke(output_dir)
    else:
        run_full(output_dir, args.epochs, args.batch_size, args.num_workers, args.allow_cpu_full)


if __name__ == "__main__":
    main()
