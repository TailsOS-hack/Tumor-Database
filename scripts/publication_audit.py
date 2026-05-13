#!/usr/bin/env python3
"""Build publication-readiness checks for the MRI classifier project.

The audit is deliberately conservative. Perfect or near-perfect CNN metrics are
not treated as proof of leakage, but they are flagged as reviewer-risk until the
split, duplicate-image, and train/validation/test gap checks are documented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_logs" / "publication_audit"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "training_logs" / "experiments"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "docs" / "colab_publication_summary.json"
MANIFEST_CANDIDATES = [
    PROJECT_ROOT / "training_logs" / "splits" / "strict_manifest.csv",
    PROJECT_ROOT / "training_logs" / "multimodal" / "kaggle_qwen_batch3" / "strict_manifest.csv",
    PROJECT_ROOT / "training_logs" / "multimodal" / "kaggle_qwen_batch2" / "strict_manifest.csv",
    PROJECT_ROOT / "training_logs" / "multimodal" / "kaggle_qwen_batch1" / "strict_manifest.csv",
]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def choose_manifest(explicit: str | None) -> Path | None:
    if explicit:
        path = resolve_repo_path(explicit)
        return path if path.exists() else None
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            return path
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path) -> str | None:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as image:
            image = image.convert("L").resize((9, 8))
            pixels = list(image.getdata())
    except Exception:
        return None

    bits: list[str] = []
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits.append("1" if pixels[offset + col] > pixels[offset + col + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def split_counter(rows: list[dict[str, str]], *keys: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        label = "::".join(row.get(key, "") for key in keys)
        counter[label] += 1
    return dict(sorted(counter.items()))


def manifest_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    duplicate_paths = [
        {"path": path, "count": count}
        for path, count in Counter(row["path"] for row in rows).items()
        if count > 1
    ]
    source_by_domain = defaultdict(Counter)
    source_by_split = defaultdict(Counter)
    for row in rows:
        source_by_domain[row["domain"]][row.get("source_split", "")] += 1
        source_by_split[row["split"]][f"{row['domain']}::{row.get('source_split', '')}"] += 1

    return {
        "rows": len(rows),
        "split_counts": split_counter(rows, "split"),
        "domain_split_counts": split_counter(rows, "split", "domain"),
        "eight_class_split_counts": split_counter(rows, "split", "eight_class"),
        "duplicate_manifest_paths": duplicate_paths,
        "source_split_by_domain": {key: dict(value) for key, value in source_by_domain.items()},
        "source_split_by_manifest_split": {key: dict(value) for key, value in source_by_split.items()},
    }


def hash_overlap_audit(rows: list[dict[str, str]], output_dir: Path, *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "reason": "Image hashing disabled by command line flag.",
            "missing_files": None,
            "files_hashed": 0,
            "exact_cross_split_overlaps": None,
            "perceptual_cross_split_overlaps": None,
        }

    exact_buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    dhash_buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_files: list[dict[str, str]] = []
    files_hashed = 0

    for row in rows:
        path = resolve_repo_path(row["path"])
        if not path.exists():
            missing_files.append({"path": row["path"], "split": row["split"], "eight_class": row["eight_class"]})
            continue
        files_hashed += 1
        metadata = {
            "path": row["path"],
            "split": row["split"],
            "domain": row["domain"],
            "eight_class": row["eight_class"],
        }
        exact_buckets[sha256_file(path)].append(metadata)
        dhash = image_dhash(path)
        if dhash:
            dhash_buckets[dhash].append(metadata)

    exact_overlaps = cross_split_rows(exact_buckets, "sha256")
    dhash_overlaps = cross_split_rows(dhash_buckets, "dhash")
    write_csv(output_dir / "exact_hash_cross_split_overlaps.csv", exact_overlaps)
    write_csv(output_dir / "perceptual_hash_cross_split_overlaps.csv", dhash_overlaps)
    write_csv(output_dir / "missing_manifest_files.csv", missing_files[:1000])

    return {
        "enabled": True,
        "files_hashed": files_hashed,
        "missing_files": len(missing_files),
        "exact_cross_split_overlaps": len(exact_overlaps),
        "perceptual_cross_split_overlaps": len(dhash_overlaps),
        "exact_overlap_csv": "exact_hash_cross_split_overlaps.csv",
        "perceptual_overlap_csv": "perceptual_hash_cross_split_overlaps.csv",
        "missing_files_csv": "missing_manifest_files.csv",
    }


def cross_split_rows(buckets: dict[str, list[dict[str, str]]], hash_name: str) -> list[dict[str, str]]:
    overlaps: list[dict[str, str]] = []
    for digest, items in buckets.items():
        splits = {item["split"] for item in items}
        if len(splits) <= 1:
            continue
        for item in items:
            overlaps.append({hash_name: digest, **item, "split_set": ",".join(sorted(splits))})
    return overlaps


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def support_from_report(report: dict[str, Any]) -> int:
    support = 0
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(value, dict):
            try:
                support += int(round(float(value.get("support", 0))))
            except (TypeError, ValueError):
                continue
    return support


def metric_value(report: dict[str, Any], key: str, metric: str) -> float | None:
    value = report.get(key, {})
    if not isinstance(value, dict):
        return None
    raw = value.get(metric)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def load_metric_rows(experiments_dir: Path, summary_json: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if summary_json and summary_json.exists():
        rows.extend(json.loads(summary_json.read_text(encoding="utf-8")))

    metric_paths = sorted(experiments_dir.glob("**/metrics.json")) if experiments_dir.exists() else []
    seen_paths = {str(row.get("metrics_path", "")) for row in rows}
    for path in metric_paths:
        rel_path = str(path.relative_to(PROJECT_ROOT))
        if rel_path in seen_paths:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        report = metrics.get("classification_report", {})
        try:
            task = path.relative_to(experiments_dir).parts[0]
        except ValueError:
            task = path.parent.name
        rows.append(
            {
                "model": task,
                "status": "complete",
                "accuracy": metrics.get("accuracy", report.get("accuracy")),
                "macro_f1": metric_value(report, "macro avg", "f1-score"),
                "weighted_f1": metric_value(report, "weighted avg", "f1-score"),
                "metrics_path": rel_path,
            }
        )
    return rows


def metrics_audit(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    audited_rows: list[dict[str, Any]] = []
    high_accuracy_flags: list[dict[str, Any]] = []
    for row in metric_rows:
        audited = dict(row)
        metrics_path = str(row.get("metrics_path", ""))
        n = 0
        if metrics_path:
            path = resolve_repo_path(metrics_path)
            if path.exists():
                metrics = json.loads(path.read_text(encoding="utf-8"))
                n = support_from_report(metrics.get("classification_report", {}))
        accuracy = as_float(row.get("accuracy"))
        if accuracy is not None and n:
            successes = int(round(accuracy * n))
            low, high = wilson_interval(successes, n)
            audited["n"] = n
            audited["accuracy_ci95_low"] = low
            audited["accuracy_ci95_high"] = high
        if accuracy is not None and accuracy >= 0.995:
            flag = {
                "model": row.get("model"),
                "accuracy": accuracy,
                "reason": "Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.",
            }
            high_accuracy_flags.append(flag)
        audited_rows.append(audited)
    return {
        "rows": audited_rows,
        "high_accuracy_flags": high_accuracy_flags,
        "metrics_available": bool(metric_rows),
    }


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def history_audit(experiments_dir: Path) -> dict[str, Any]:
    histories: list[dict[str, Any]] = []
    for path in sorted(experiments_dir.glob("**/history.json")) if experiments_dir.exists() else []:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not rows:
            continue
        task = path.parent.parent.name if path.parent.parent != experiments_dir else path.parent.name
        gaps = [
            as_float(row.get("train_accuracy")) - as_float(row.get("val_accuracy"))
            for row in rows
            if as_float(row.get("train_accuracy")) is not None and as_float(row.get("val_accuracy")) is not None
        ]
        if not gaps:
            continue
        max_gap = max(gaps)
        best_val = max(as_float(row.get("val_accuracy")) or 0.0 for row in rows)
        final = rows[-1]
        histories.append(
            {
                "task": task,
                "history_path": str(path.relative_to(PROJECT_ROOT)),
                "epochs_recorded": len(rows),
                "best_val_accuracy": best_val,
                "final_train_accuracy": final.get("train_accuracy"),
                "final_val_accuracy": final.get("val_accuracy"),
                "max_train_val_accuracy_gap": max_gap,
                "flag": "gap_gt_0.05" if max_gap > 0.05 else "",
            }
        )
    return {
        "histories": histories,
        "histories_available": bool(histories),
        "gap_flags": [row for row in histories if row.get("flag")],
    }


def risk_summary(
    manifest: dict[str, Any],
    hash_audit: dict[str, Any],
    metrics: dict[str, Any],
    histories: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    if manifest.get("duplicate_manifest_paths"):
        blockers.append("Duplicate manifest paths exist.")
    if not hash_audit.get("enabled", False):
        warnings.append("Image hash duplicate checks were skipped.")
    if (hash_audit.get("exact_cross_split_overlaps") or 0) > 0:
        blockers.append("Exact image hash overlap across train/val/test splits.")
    if (hash_audit.get("perceptual_cross_split_overlaps") or 0) > 0:
        warnings.append("Perceptual hash overlap across splits needs manual review.")
    if (hash_audit.get("missing_files") or 0) > 0:
        warnings.append("Some manifest files were missing during the audit.")
    if metrics.get("high_accuracy_flags"):
        warnings.append("Near-perfect metrics require leakage/source-bias and external-validity discussion.")
    if not histories.get("histories_available"):
        warnings.append("Training histories were not available, so overfitting gaps could not be audited locally.")
    if histories.get("gap_flags"):
        warnings.append("One or more histories show train/validation accuracy gap > 0.05.")

    source_by_domain = manifest.get("source_split_by_domain", {})
    if source_by_domain.get("tumor") and source_by_domain.get("dementia"):
        warnings.append(
            "Binary router may learn dataset/domain artifacts because tumor and dementia images come from different source datasets."
        )

    if blockers:
        status = "blocking_leakage_risk"
    elif warnings:
        status = "reviewer_risk_needs_documentation"
    else:
        status = "no_obvious_leakage_detected"

    return {"status": status, "blockers": blockers, "warnings": warnings}


def write_report(audit: dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "audit_report.md"
    lines = [
        "# Publication Audit Report",
        "",
        f"Overall status: `{audit['risk_summary']['status']}`",
        "",
        "## Leakage Checks",
        "",
        f"- Manifest rows: {audit['manifest'].get('rows', 0)}",
        f"- Duplicate manifest paths: {len(audit['manifest'].get('duplicate_manifest_paths', []))}",
        f"- Exact cross-split hash overlaps: {format_count(audit['image_hashes'].get('exact_cross_split_overlaps'))}",
        f"- Perceptual cross-split hash overlaps: {format_count(audit['image_hashes'].get('perceptual_cross_split_overlaps'))}",
        f"- Missing manifest files: {format_count(audit['image_hashes'].get('missing_files'))}",
        "",
        "## Metric Risk Flags",
        "",
    ]
    flags = audit["metrics"].get("high_accuracy_flags", [])
    if flags:
        for flag in flags:
            lines.append(f"- {flag['model']}: accuracy {flag['accuracy']:.4f}. {flag['reason']}")
    else:
        lines.append("- No near-perfect metric flags found in available summaries.")

    lines.extend(["", "## Overfitting History", ""])
    histories = audit["histories"].get("histories", [])
    if histories:
        lines.append("| Task | Epochs | Best Val Acc | Final Train Acc | Final Val Acc | Max Gap | Flag |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in histories:
            lines.append(
                "| {task} | {epochs_recorded} | {best_val_accuracy:.4f} | {final_train_accuracy} | "
                "{final_val_accuracy} | {max_train_val_accuracy_gap:.4f} | {flag} |".format(**row)
            )
    else:
        lines.append("- Training histories were not available in the inspected experiments directory.")

    lines.extend(["", "## Warnings", ""])
    warnings = audit["risk_summary"].get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def format_count(value: Any) -> str:
    return "not run" if value is None else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Strict manifest CSV. Defaults to the newest available strict manifest.")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-image-hashes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    experiments_dir = resolve_repo_path(args.experiments_dir)
    summary_json = resolve_repo_path(args.summary_json) if args.summary_json else None
    manifest_path = choose_manifest(args.manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path:
        raise SystemExit("No strict manifest found. Run create-manifest first or pass --manifest.")

    rows = read_csv(manifest_path)
    manifest = manifest_audit(rows)
    manifest["manifest_path"] = str(manifest_path.relative_to(PROJECT_ROOT))
    image_hashes = hash_overlap_audit(rows, output_dir, enabled=not args.skip_image_hashes)
    metric_rows = load_metric_rows(experiments_dir, summary_json)
    metrics = metrics_audit(metric_rows)
    histories = history_audit(experiments_dir)
    audit = {
        "manifest": manifest,
        "image_hashes": image_hashes,
        "metrics": metrics,
        "histories": histories,
        "risk_summary": risk_summary(manifest, image_hashes, metrics, histories),
    }

    (output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report_path = write_report(audit, output_dir)
    print(f"Wrote {report_path}")
    print(json.dumps(audit["risk_summary"], indent=2))


if __name__ == "__main__":
    main()
