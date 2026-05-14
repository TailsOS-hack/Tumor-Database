#!/usr/bin/env python3
"""Build manuscript-ready result tables from committed experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

PRIMARY_SUMMARY = DOCS_DIR / "dedup_publication_summary.json"
SENSITIVITY_SUMMARY = DOCS_DIR / "perceptual_sensitivity_summary.json"
INITIAL_AUDIT = PROJECT_ROOT / "training_logs" / "publication_audit" / "cnn_publication_audit_summary.json"
DEDUP_AUDIT = PROJECT_ROOT / "training_logs" / "publication_audit" / "cnn_dedup_retrain_summary.json"
PERCEPTUAL_AUDIT = (
    PROJECT_ROOT / "training_logs" / "publication_audit" / "cnn_perceptual_sensitivity_summary.json"
)

OUTPUT_MD = DOCS_DIR / "PUBLICATION_RESULTS_TABLES.md"
CNN_CSV = DOCS_DIR / "publication_cnn_results.csv"
AUDIT_CSV = DOCS_DIR / "publication_audit_checks.csv"
VLM_CSV = DOCS_DIR / "publication_vlm_results.csv"

MODEL_LABELS = {
    "binary": "Binary router",
    "tumor": "Tumor specialist",
    "dementia": "Dementia specialist",
    "hierarchical": "Hierarchical CNN",
    "eight_class": "Single 8-class CNN",
}

VLM_ROWS = [
    {
        "family": "VLM zero-shot",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "n": 40,
        "json_rate": 1.0,
        "accuracy": 0.1750,
        "note": "Best flat zero-shot VLM from batch 2; not competitive with CNNs.",
    },
    {
        "family": "VLM LoRA",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct + LoRA",
        "n": 96,
        "json_rate": 1.0,
        "accuracy": 0.3125,
        "note": "Improved direct VLM labeling but collapsed dementia subclasses.",
    },
    {
        "family": "VLM hierarchical diagnostic",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "n": 40,
        "json_rate": 1.0,
        "accuracy": 0.2250,
        "domain_accuracy": 0.8750,
        "oracle_domain_subtype_accuracy": 0.2500,
        "note": "Broad-domain routing improved; subtype labels remained weak.",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(path: str, key: str) -> Any:
    metrics_path = PROJECT_ROOT / path
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    return metrics.get(key)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{int(value):,}"


def enrich_rows(summary_path: Path, cohort: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_json(summary_path):
        metrics_path = str(row.get("metrics_path", ""))
        n = metric_value(metrics_path, "n")
        correct = metric_value(metrics_path, "correct")
        ci_low = ci_high = None
        if isinstance(n, int) and isinstance(correct, int):
            ci_low, ci_high = wilson_interval(correct, n)
        rows.append(
            {
                "cohort": cohort,
                "model": row["model"],
                "label": MODEL_LABELS.get(str(row["model"]), str(row["model"])),
                "n": n,
                "accuracy": row.get("accuracy"),
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "macro_f1": row.get("macro_f1"),
                "weighted_f1": row.get("weighted_f1"),
                "metrics_path": metrics_path,
            }
        )
    return rows


def audit_row(name: str, summary_path: Path, audit_key: str) -> dict[str, Any]:
    summary = load_json(summary_path)
    audit = summary[audit_key]
    image_hashes = audit["image_hashes"]
    manifest = audit["manifest"]
    histories = audit.get("histories", {})
    return {
        "analysis": name,
        "manifest_rows": manifest.get("rows"),
        "train": manifest.get("split_counts", {}).get("train"),
        "val": manifest.get("split_counts", {}).get("val"),
        "test": manifest.get("split_counts", {}).get("test"),
        "exact_hash_overlaps": image_hashes.get("exact_cross_split_overlaps"),
        "perceptual_hash_overlaps": image_hashes.get("perceptual_cross_split_overlaps"),
        "missing_files": image_hashes.get("missing_files"),
        "gap_flags": len(histories.get("gap_flags", [])),
        "risk_status": audit.get("risk_summary", {}).get("status"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_markdown(cnn_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> str:
    primary = [row for row in cnn_rows if row["cohort"] == "Primary exact-deduplicated"]
    sensitivity = [row for row in cnn_rows if row["cohort"] == "Conservative perceptual sensitivity"]

    def cnn_md_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                row["label"],
                fmt_int(row["n"]),
                fmt(row["accuracy"]),
                f"{fmt(row['accuracy_ci95_low'])}-{fmt(row['accuracy_ci95_high'])}",
                fmt(row["macro_f1"]),
                fmt(row["weighted_f1"]),
                f"`{row['metrics_path']}`",
            ]
            for row in rows
        ]

    lines: list[str] = [
        "# Publication Results Tables",
        "",
        "Generated by `scripts/build_publication_tables.py` from committed metrics and audit JSON artifacts.",
        "",
        "## Primary CNN Results",
        "",
        "These are the accepted exact-deduplicated checkpoints currently stored in `models/*.pt`.",
        "",
        *markdown_table(
            ["Model", "N", "Accuracy", "95% CI", "Macro F1", "Weighted F1", "Metrics"],
            cnn_md_rows(primary),
        ),
        "",
        "## Conservative Perceptual Sensitivity",
        "",
        "This run groups exact SHA-256 duplicates and identical audit-compatible dHash fingerprints into the same split. It is a robustness analysis, not the default checkpoint set.",
        "",
        *markdown_table(
            ["Model", "N", "Accuracy", "95% CI", "Macro F1", "Weighted F1", "Metrics"],
            cnn_md_rows(sensitivity),
        ),
        "",
        "## Leakage And Robustness Checks",
        "",
        *markdown_table(
            [
                "Analysis",
                "Rows",
                "Train",
                "Val",
                "Test",
                "Exact Overlaps",
                "dHash Overlaps",
                "Missing",
                "Gap Flags",
                "Risk Status",
            ],
            [
                [
                    row["analysis"],
                    fmt_int(row["manifest_rows"]),
                    fmt_int(row["train"]),
                    fmt_int(row["val"]),
                    fmt_int(row["test"]),
                    fmt_int(row["exact_hash_overlaps"]),
                    fmt_int(row["perceptual_hash_overlaps"]),
                    fmt_int(row["missing_files"]),
                    fmt_int(row["gap_flags"]),
                    str(row["risk_status"]),
                ]
                for row in audit_rows
            ],
        ),
        "",
        "## Multimodal VLM Results",
        "",
        "These results justify keeping the CNNs as the diagnostic image classifiers and treating VLMs as experimental report/metadata helpers.",
        "",
        *markdown_table(
            ["Family", "Model", "N", "JSON Rate", "Accuracy", "Domain Acc.", "Oracle Subtype Acc.", "Note"],
            [
                [
                    row["family"],
                    row["model"],
                    fmt_int(row["n"]),
                    fmt(row.get("json_rate")),
                    fmt(row.get("accuracy")),
                    fmt(row.get("domain_accuracy")),
                    fmt(row.get("oracle_domain_subtype_accuracy")),
                    row["note"],
                ]
                for row in VLM_ROWS
            ],
        ),
        "",
        "## Manuscript Claim Boundaries",
        "",
        "- Primary claim: the exact-deduplicated CNN suite achieves strong strict-test performance on this MRI dataset.",
        "- Robustness claim: the conservative dHash sensitivity run keeps performance high after removing both exact and identical-dHash cross-split overlap.",
        "- Architecture comparison: the single 8-class CNN is slightly higher in accuracy than the hierarchical CNN in both accepted and sensitivity runs.",
        "- Limitation: binary tumor-vs-dementia routing may exploit dataset/source differences because tumor and dementia images come from different source datasets.",
        "- Limitation: external validation on an independent MRI cohort is still required before clinical claims.",
        "- Multimodal conclusion: VLMs are not competitive as direct MRI classifiers in these experiments.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cnn_rows = enrich_rows(PRIMARY_SUMMARY, "Primary exact-deduplicated")
    cnn_rows.extend(enrich_rows(SENSITIVITY_SUMMARY, "Conservative perceptual sensitivity"))
    audit_rows = [
        audit_row("Initial leaky audit", INITIAL_AUDIT, "pre_regularized_audit"),
        audit_row("Accepted exact-deduplicated", DEDUP_AUDIT, "regularized_audit"),
        audit_row("Conservative dHash sensitivity", PERCEPTUAL_AUDIT, "regularized_audit"),
    ]

    write_csv(CNN_CSV, cnn_rows)
    write_csv(AUDIT_CSV, audit_rows)
    write_csv(VLM_CSV, VLM_ROWS)
    OUTPUT_MD.write_text(build_markdown(cnn_rows, audit_rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {CNN_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {AUDIT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {VLM_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
