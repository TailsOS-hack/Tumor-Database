#!/usr/bin/env python3
"""Import a Colab training export zip into the local Tumor-Database repo.

This script copies the trained checkpoints and result summaries into the repo,
sets up Git LFS tracking for model files, and cleans its temporary extraction
directory. It does not commit or push by itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_ROOT = Path("/private/tmp/tumor-db-colab-import")
MODEL_FILES = [
    "models/binary_router.pt",
    "models/brain_tumor_classifier.pt",
    "models/alzheimers_classifier.pt",
    "models/single_8class_classifier.pt",
]


def run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def safe_extract(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (target_dir / member.filename).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe archive path: {member.filename}") from exc
        archive.extractall(target_dir)


def find_export_root(staging_dir: Path) -> Path:
    direct = staging_dir / "colab_export_manifest.json"
    if direct.exists():
        return staging_dir
    matches = list(staging_dir.rglob("colab_export_manifest.json"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one colab_export_manifest.json, found {len(matches)} in {staging_dir}")
    return matches[0].parent


def read_manifest(export_root: Path) -> dict[str, object]:
    manifest_path = export_root / "colab_export_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {source} -> {destination}", flush=True)


def copy_tree_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Copied {source} -> {destination}", flush=True)


def ensure_git_lfs(repo_root: Path, skip_lfs: bool) -> None:
    if skip_lfs:
        print("Skipping Git LFS setup because --skip-lfs was provided.", flush=True)
        return

    version = run(["git", "lfs", "version"], cwd=repo_root, check=False)
    if version.returncode != 0:
        raise SystemExit(
            "Git LFS is not installed or not available in PATH. Install Git LFS, then rerun this importer."
        )

    run(["git", "lfs", "install", "--local"], cwd=repo_root)
    run(["git", "lfs", "track", "models/*.pt"], cwd=repo_root)


def import_export(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    archive_path = Path(args.archive).expanduser().resolve()
    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")

    staging_dir = Path(args.staging_root).resolve()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    safe_extract(archive_path, staging_dir)
    export_root = find_export_root(staging_dir)
    manifest = read_manifest(export_root)
    mode = str(manifest.get("mode", "unknown"))
    if mode != "full" and not args.allow_smoke:
        raise SystemExit(
            f"Refusing to import a non-full export (mode={mode}). "
            "Use --allow-smoke only for local testing."
        )

    missing = [rel for rel in MODEL_FILES if not (export_root / rel).exists()]
    if missing:
        raise SystemExit("Export is missing expected model files: " + ", ".join(missing))

    ensure_git_lfs(repo_root, args.skip_lfs)

    for rel in MODEL_FILES:
        copy_file(export_root / rel, repo_root / rel)

    if args.copy_logs:
        copy_tree_if_exists(export_root / "training_logs" / "splits", repo_root / "training_logs" / "splits")
        copy_tree_if_exists(export_root / "training_logs" / "experiments", repo_root / "training_logs" / "experiments")

    summary_md = export_root / "training_logs" / "experiments" / "publication_summary.md"
    summary_json = export_root / "training_logs" / "experiments" / "publication_summary.json"
    if summary_md.exists():
        copy_file(summary_md, repo_root / "docs" / "COLAB_PUBLICATION_RESULTS.md")
    if summary_json.exists():
        copy_file(summary_json, repo_root / "docs" / "colab_publication_summary.json")

    (repo_root / "docs" / "colab_export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(staging_dir)
    print(f"Deleted temporary extraction directory: {staging_dir}", flush=True)

    if args.delete_archive:
        archive_path.unlink()
        print(f"Deleted downloaded archive: {archive_path}", flush=True)

    print(
        "\nImport complete. Next checks:\n"
        "  git lfs ls-files\n"
        "  git status --short\n"
        "Then commit and push the model update.",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Downloaded tumor_database_colab_artifacts_*.zip")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-lfs", action="store_true")
    parser.add_argument("--copy-logs", dest="copy_logs", action="store_true", default=True)
    parser.add_argument("--no-copy-logs", dest="copy_logs", action="store_false")
    parser.add_argument("--delete-archive", action="store_true")
    return parser


def main() -> None:
    import_export(build_parser().parse_args())


if __name__ == "__main__":
    main()
