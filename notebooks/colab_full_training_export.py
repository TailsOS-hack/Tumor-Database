#!/usr/bin/env python3
"""One-shot Google Colab runner for the full training/export flow.

Recommended Colab usage:

    !python /content/colab_full_training_export.py --epochs 30 --batch-size 32

The script clones the repo into Colab, runs the strict training suite, packages
the trained model files plus metrics into one zip, copies that zip to Drive,
and asks Colab to download it through the browser.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path


REPO_URL = "https://github.com/TailsOS-hack/Tumor-Database.git"
DEFAULT_WORK_DIR = Path("/content/Tumor-Database")
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/Tumor-Database")
MODEL_FILES = [
    "models/binary_router.pt",
    "models/brain_tumor_classifier.pt",
    "models/alzheimers_classifier.pt",
    "models/single_8class_classifier.pt",
]
DOC_FILES = [
    "README.md",
    "docs/ML_EXECUTION_FLOW.md",
    "docs/PUBLICATION_NOTES.md",
]
ARTIFACT_DIRS = [
    "training_logs/splits",
    "training_logs/experiments",
]
ARCHIVE_PROGRESS_LOG = "colab_run_progress.jsonl"
ARCHIVE_CONSOLE_LOG = "colab_console.log"


def quote_command(command: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def console_log_path() -> Path | None:
    value = os.environ.get("TUMOR_DB_CONSOLE_LOG", "")
    return Path(value) if value else None


def append_console_log(text: str) -> None:
    path = console_log_path()
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    print(f"$ {quote_command(command)}", flush=True)
    append_console_log(f"$ {quote_command(command)}\n")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        append_console_log(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, [str(part) for part in command])


def progress_log_path(args: argparse.Namespace) -> Path:
    return Path(args.progress_log)


def log_progress(args: argparse.Namespace, step: str, status: str, detail: str = "") -> None:
    timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    message = f"[{timestamp}] {status}: {step}"
    if detail:
        message += f" ({detail})"
    print("\n" + "=" * 80, flush=True)
    print(message, flush=True)
    print("=" * 80, flush=True)
    append_console_log("\n" + "=" * 80 + "\n")
    append_console_log(message + "\n")
    append_console_log("=" * 80 + "\n")

    path = progress_log_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "created_utc": timestamp,
                    "step": step,
                    "status": status,
                    "detail": detail,
                }
            )
            + "\n"
        )


@contextmanager
def progress_step(args: argparse.Namespace, step: str):
    started = time.monotonic()
    log_progress(args, step, "START")
    try:
        yield
    except Exception as exc:
        elapsed = f"{time.monotonic() - started:.1f}s"
        log_progress(args, step, "FAILED", f"{type(exc).__name__}: {exc}; elapsed={elapsed}")
        raise
    else:
        elapsed = f"{time.monotonic() - started:.1f}s"
        log_progress(args, step, "DONE", f"elapsed={elapsed}")


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def mount_drive() -> None:
    if not in_colab():
        print("Not running in Colab; skipping Drive mount.", flush=True)
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith("/content/") or resolved == Path("/content"):
        raise RuntimeError(f"Refusing to delete unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def clone_repo(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir)
    if args.fresh_clone and work_dir.exists():
        safe_remove_tree(work_dir)

    if work_dir.exists() and (work_dir / ".git").exists():
        run(["git", "fetch", "--depth", "1", "origin", args.branch], cwd=work_dir)
        run(["git", "checkout", args.branch], cwd=work_dir)
        run(["git", "reset", "--hard", f"origin/{args.branch}"], cwd=work_dir)
        return work_dir

    work_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--branch", args.branch, args.repo_url, work_dir])
    return work_dir


def install_dependencies(repo_root: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pillow",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "pandas",
            "tqdm",
        ],
        cwd=repo_root,
    )


def show_gpu() -> None:
    try:
        run(["nvidia-smi"])
    except subprocess.CalledProcessError:
        print("nvidia-smi failed or no GPU is attached.", flush=True)


def run_training_suite(repo_root: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--mode",
        args.mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        "training_logs/experiments",
    ]
    if args.allow_cpu_full:
        command.append("--allow-cpu-full")
    run(command, cwd=repo_root)


def git_value(repo_root: Path, command: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_artifact_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in MODEL_FILES + DOC_FILES:
        path = repo_root / rel
        if path.exists():
            files.append(path)

    for rel_dir in ARTIFACT_DIRS:
        path = repo_root / rel_dir
        if path.exists():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    return sorted(set(files))


def collect_model_metadata(repo_root: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for rel in MODEL_FILES:
        path = repo_root / rel
        if path.exists():
            metadata[rel] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return metadata


def package_artifacts(repo_root: Path, args: argparse.Namespace) -> Path:
    missing = [rel for rel in MODEL_FILES if not (repo_root / rel).exists()]
    if args.mode == "full" and missing:
        raise SystemExit(
            "Full run finished without all expected model files. Missing: "
            + ", ".join(missing)
        )

    export_root = Path(args.export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    archive_path = export_root / f"tumor_database_colab_artifacts_{timestamp}.zip"

    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "repo_url": args.repo_url,
        "branch": args.branch,
        "commit": git_value(repo_root, ["rev-parse", "HEAD"]),
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "progress_log": ARCHIVE_PROGRESS_LOG,
        "console_log": ARCHIVE_CONSOLE_LOG,
        "model_files": collect_model_metadata(repo_root),
        "archive_name": archive_path.name,
    }

    files = iter_artifact_files(repo_root)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("colab_export_manifest.json", json.dumps(manifest, indent=2))
        progress_path = progress_log_path(args)
        if progress_path.exists():
            archive.write(progress_path, ARCHIVE_PROGRESS_LOG)
        console_path = console_log_path()
        if console_path and console_path.exists():
            archive.write(console_path, ARCHIVE_CONSOLE_LOG)
        for path in files:
            archive.write(path, path.relative_to(repo_root).as_posix())

    print(f"Packaged {len(files)} files into {archive_path}", flush=True)
    return archive_path


def download_archive(archive_path: Path) -> None:
    if not in_colab():
        print(f"Archive ready: {archive_path}", flush=True)
        return
    from google.colab import files  # type: ignore

    print("Starting browser download. Keep this tab open until the download starts.", flush=True)
    files.download(str(archive_path))


def package_failure(args: argparse.Namespace, error: BaseException) -> Path:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    archive_path = Path("/content") / f"tumor_database_colab_failure_{timestamp}.zip"
    failure_payload = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "error_type": type(error).__name__,
        "error": str(error),
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "next_steps": [
            "Scroll above to the FAILED banner and the command output immediately before it.",
            "If the error says CUDA/GPU is unavailable, switch Colab to Runtime > Change runtime type > GPU and rerun.",
            "Send this failure zip or the last 120 console-log lines back for diagnosis.",
        ],
    }

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("colab_failure.json", json.dumps(failure_payload, indent=2))
        progress_path = progress_log_path(args)
        if progress_path.exists():
            archive.write(progress_path, ARCHIVE_PROGRESS_LOG)
        console_path = console_log_path()
        if console_path and console_path.exists():
            archive.write(console_path, ARCHIVE_CONSOLE_LOG)

    try:
        export_root = Path(args.export_root)
        export_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_path, export_root / archive_path.name)
    except Exception as copy_error:
        print(f"Could not copy failure zip to Drive/export folder: {copy_error}", flush=True)

    print(f"Failure bundle ready: {archive_path}", flush=True)
    if in_colab():
        try:
            from google.colab import files  # type: ignore

            files.download(str(archive_path))
        except Exception as download_error:
            print(f"Could not start failure-bundle browser download: {download_error}", flush=True)
    return archive_path


def print_console_tail(line_count: int = 160) -> None:
    path = console_log_path()
    if not path or not path.exists():
        print("No console log was found to print.", flush=True)
        return

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n" + "=" * 80, flush=True)
    print(f"Last {min(line_count, len(lines))} lines from {path}", flush=True)
    print("=" * 80, flush=True)
    for line in lines[-line_count:]:
        print(line, flush=True)


def cleanup_colab_clone(repo_root: Path, archive_path: Path, enabled: bool) -> None:
    if not enabled:
        return
    try:
        archive_path.resolve().relative_to(repo_root.resolve())
        print("Archive is inside the repo clone; skipping clone cleanup.", flush=True)
        return
    except ValueError:
        pass
    safe_remove_tree(repo_root)
    print(f"Deleted Colab temp clone: {repo_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=os.environ.get("TUMOR_DB_REPO_URL", REPO_URL))
    parser.add_argument("--branch", default=os.environ.get("TUMOR_DB_BRANCH", "main"))
    parser.add_argument("--work-dir", default=os.environ.get("TUMOR_DB_WORK_DIR", str(DEFAULT_WORK_DIR)))
    parser.add_argument(
        "--drive-root",
        default=os.environ.get("TUMOR_DB_DRIVE_ROOT", str(DEFAULT_DRIVE_ROOT)),
        help="Drive folder used for durable Colab exports.",
    )
    parser.add_argument("--export-root", default=os.environ.get("TUMOR_DB_EXPORT_ROOT", ""))
    parser.add_argument("--progress-log", default=os.environ.get("TUMOR_DB_PROGRESS_LOG", ""))
    parser.add_argument("--console-log", default=os.environ.get("TUMOR_DB_CONSOLE_LOG", ""))
    parser.add_argument("--mode", choices=["smoke", "full"], default=os.environ.get("EXPERIMENT_MODE", "full"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EXPERIMENT_EPOCHS", "30")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("EXPERIMENT_BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("EXPERIMENT_NUM_WORKERS", "0")))
    parser.add_argument("--allow-cpu-full", action="store_true")
    parser.add_argument("--fresh-clone", dest="fresh_clone", action="store_true", default=True)
    parser.add_argument("--reuse-clone", dest="fresh_clone", action="store_false")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-drive-mount", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false", default=True)
    parser.add_argument("--cleanup-clone", dest="cleanup_clone", action="store_true", default=True)
    parser.add_argument("--keep-clone", dest="cleanup_clone", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.export_root:
        args.export_root = str(Path(args.drive_root) / "exports")
    if not args.progress_log:
        args.progress_log = str(Path(args.work_dir).parent / "tumor_database_colab_progress.jsonl")
    if not args.console_log:
        args.console_log = str(Path(args.work_dir).parent / "tumor_database_colab_console.log")
    os.environ["TUMOR_DB_CONSOLE_LOG"] = args.console_log

    progress_path = progress_log_path(args)
    if progress_path.exists():
        progress_path.unlink()
    console_path = console_log_path()
    if console_path and console_path.exists():
        console_path.unlink()

    log_progress(args, "Colab full training export", "START", f"mode={args.mode}, epochs={args.epochs}")

    try:
        with progress_step(args, "1/8 Mount Google Drive"):
            if not args.skip_drive_mount:
                mount_drive()
            else:
                print("Skipping Drive mount because --skip-drive-mount was provided.", flush=True)

        with progress_step(args, "2/8 Fresh clone Tumor-Database"):
            repo_root = clone_repo(args)

        with progress_step(args, "3/8 Install training dependencies"):
            if not args.skip_install:
                install_dependencies(repo_root)
            else:
                print("Skipping dependency install because --skip-install was provided.", flush=True)

        with progress_step(args, "4/8 Check GPU"):
            show_gpu()

        with progress_step(args, "5/8 Run strict training suite"):
            run_training_suite(repo_root, args)

        with progress_step(args, "6/8 Package models and metrics"):
            archive_path = package_artifacts(repo_root, args)

        with progress_step(args, "7/8 Start browser download"):
            if args.download:
                download_archive(archive_path)
            else:
                print("Skipping browser download because --no-download was provided.", flush=True)

        with progress_step(args, "8/8 Delete temporary Colab clone"):
            cleanup_colab_clone(repo_root, archive_path, args.cleanup_clone)

        log_progress(args, "Colab full training export", "DONE", f"archive={archive_path}")
        print("Done. Send the downloaded zip back to the local machine for import.", flush=True)
    except Exception as exc:
        log_progress(args, "Colab full training export", "FAILED", f"{type(exc).__name__}: {exc}")
        package_failure(args, exc)
        print_console_tail()
        print(
            "\nRun failed. If this was a GPU/CUDA error, set Colab to Runtime > Change runtime type > GPU "
            "and rerun the same cell. Otherwise send me the downloaded failure zip.",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
