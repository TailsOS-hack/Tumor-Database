#!/usr/bin/env python3
"""Validate and summarize the publication package.

This is intentionally lightweight: it only checks committed tables, metrics,
figures, scripts, and documentation. It does not retrain models or touch large
datasets.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_PATH = DOCS_DIR / "PUBLICATION_SUBMISSION_STATUS.md"


REQUIRED_PATHS = {
    "manuscript_docs": [
        "docs/DATASET_PROVENANCE.md",
        "docs/MANUSCRIPT_DRAFT.md",
        "docs/MANUSCRIPT_FULL_DRAFT.md",
        "docs/PUBLICATION_RESULTS_TABLES.md",
        "docs/PUBLICATION_EVIDENCE_RESULTS.md",
        "docs/PUBLICATION_FIGURES.md",
        "docs/FIGURE_CAPTIONS.md",
        "docs/GROUNDED_REPORTING.md",
    ],
    "tables": [
        "docs/publication_cnn_results.csv",
        "docs/publication_audit_checks.csv",
        "docs/publication_vlm_results.csv",
    ],
    "figures": [
        "docs/figures/figure1_workflow.png",
        "docs/figures/figure2_architecture.png",
        "docs/figures/figure3_exact_dedup_confusion.png",
        "docs/figures/figure4_dhash_sensitivity_confusion.png",
        "docs/figures/figure5_cnn_vlm_comparison.png",
        "docs/figures/figure6_calibration_confidence.png",
        "docs/figures/figure7_roc_pr_curves.png",
    ],
    "evidence": [
        "training_logs/publication_evidence/publication_evidence_summary.json",
        "training_logs/publication_audit/cnn_dedup_retrain_summary.json",
        "training_logs/publication_audit/cnn_perceptual_sensitivity_summary.json",
        "training_logs/experiments_dedup_regularized/publication_summary.json",
        "training_logs/experiments_perceptual_regularized/publication_summary.json",
    ],
    "source_scripts": [
        "scripts/build_publication_tables.py",
        "scripts/build_publication_evidence.py",
        "scripts/build_publication_figures.py",
        "scripts/check_publication_package.py",
        "scripts/publication_audit.py",
        "src/grounded_report.py",
        "src/hierarchical_inference.py",
    ],
}


PRIMARY_MODELS = {
    "binary": "Binary router",
    "tumor": "Tumor specialist",
    "dementia": "Dementia specialist",
    "hierarchical": "Hierarchical CNN",
    "eight_class": "Single 8-class CNN",
}


EVIDENCE_LABELS = {
    "binary": "Binary router",
    "tumor": "Tumor specialist",
    "dementia": "Dementia specialist",
    "eight_class": "Single 8-class CNN",
    "hierarchical": "Hierarchical CNN",
}


@dataclass
class Check:
    status: str
    item: str
    evidence: str
    action: str = ""


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_csv(path: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str):
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", "NA", "None", None}:
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in {"", "NA", "None", None}:
        return default
    return int(float(value))


def pct(value: float) -> str:
    return f"{value:.4f}"


def md_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def check_required_paths() -> list[Check]:
    checks: list[Check] = []
    for section, paths in REQUIRED_PATHS.items():
        missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]
        status = "PASS" if not missing else "FAIL"
        evidence = f"{len(paths) - len(missing)}/{len(paths)} files present"
        action = "" if not missing else "Missing: " + ", ".join(f"`{path}`" for path in missing)
        checks.append(Check(status, section.replace("_", " ").title(), evidence, action))
    return checks


def check_cnn_results(rows: list[dict[str, str]]) -> tuple[list[Check], list[dict[str, str]], list[dict[str, str]]]:
    checks: list[Check] = []
    primary = [row for row in rows if row["cohort"] == "Primary exact-deduplicated"]
    sensitivity = [row for row in rows if row["cohort"] == "Conservative perceptual sensitivity"]
    primary_models = {row["model"] for row in primary}
    sensitivity_models = {row["model"] for row in sensitivity}

    checks.append(
        Check(
            "PASS" if set(PRIMARY_MODELS) <= primary_models else "FAIL",
            "Primary CNN rows",
            f"{len(primary)}/5 primary rows found",
            "" if set(PRIMARY_MODELS) <= primary_models else "Regenerate `docs/publication_cnn_results.csv`.",
        )
    )
    checks.append(
        Check(
            "PASS" if set(PRIMARY_MODELS) <= sensitivity_models else "FAIL",
            "Perceptual sensitivity rows",
            f"{len(sensitivity)}/5 sensitivity rows found",
            "" if set(PRIMARY_MODELS) <= sensitivity_models else "Regenerate `docs/publication_cnn_results.csv`.",
        )
    )

    primary_by_model = {row["model"]: row for row in primary}
    hierarchical = as_float(primary_by_model.get("hierarchical", {}), "accuracy")
    single = as_float(primary_by_model.get("eight_class", {}), "accuracy")
    checks.append(
        Check(
            "PASS" if single >= 0.99 and hierarchical >= 0.99 else "FAIL",
            "Primary CNN performance",
            f"single_8class={pct(single)}, hierarchical={pct(hierarchical)}",
            "Investigate model artifacts before manuscript submission." if min(single, hierarchical) < 0.99 else "",
        )
    )

    tumor = as_float(primary_by_model.get("tumor", {}), "accuracy")
    dementia = as_float(primary_by_model.get("dementia", {}), "accuracy")
    checks.append(
        Check(
            "PASS" if tumor >= 0.95 and dementia >= 0.99 else "FAIL",
            "Specialist performance",
            f"tumor={pct(tumor)}, dementia={pct(dementia)}",
            "Keep specialist limitations explicit." if tumor < 0.95 or dementia < 0.99 else "",
        )
    )

    return checks, primary, sensitivity


def check_audits(rows: list[dict[str, str]]) -> list[Check]:
    checks: list[Check] = []
    by_name = {row["analysis"]: row for row in rows}
    initial = by_name.get("Initial leaky audit", {})
    accepted = by_name.get("Accepted exact-deduplicated", {})
    sensitivity = by_name.get("Conservative dHash sensitivity", {})

    checks.append(
        Check(
            "PASS" if as_int(initial, "exact_hash_overlaps") > 0 else "WARN",
            "Initial leakage audit retained",
            f"initial exact overlaps={as_int(initial, 'exact_hash_overlaps')}",
            "Explain this as methodological correction, not final performance." if as_int(initial, "exact_hash_overlaps") > 0 else "Confirm initial audit artifact exists.",
        )
    )
    checks.append(
        Check(
            "PASS"
            if as_int(accepted, "exact_hash_overlaps") == 0 and as_int(accepted, "missing_files") == 0
            else "FAIL",
            "Accepted exact-deduplicated audit",
            f"exact overlaps={as_int(accepted, 'exact_hash_overlaps')}, missing={as_int(accepted, 'missing_files')}",
            "Do not submit primary claims until exact leakage is fixed."
            if as_int(accepted, "exact_hash_overlaps") != 0 or as_int(accepted, "missing_files") != 0
            else "",
        )
    )
    checks.append(
        Check(
            "PASS"
            if as_int(sensitivity, "exact_hash_overlaps") == 0
            and as_int(sensitivity, "perceptual_hash_overlaps") == 0
            and as_int(sensitivity, "missing_files") == 0
            else "FAIL",
            "Conservative dHash sensitivity audit",
            (
                f"exact={as_int(sensitivity, 'exact_hash_overlaps')}, "
                f"dHash={as_int(sensitivity, 'perceptual_hash_overlaps')}, "
                f"missing={as_int(sensitivity, 'missing_files')}"
            ),
            "Keep dHash sensitivity run as robustness evidence.",
        )
    )
    return checks


def check_evidence(rows: list[dict]) -> tuple[list[Check], list[dict]]:
    checks: list[Check] = []
    by_task = {row["task"]: row for row in rows}
    missing = sorted(set(EVIDENCE_LABELS) - set(by_task))
    checks.append(
        Check(
            "PASS" if not missing else "FAIL",
            "Probability-level evidence",
            f"{len(by_task)}/5 evidence summaries found",
            "" if not missing else "Missing evidence summaries for: " + ", ".join(missing),
        )
    )

    calibration_warnings = []
    for task in ["tumor", "dementia", "eight_class", "hierarchical"]:
        ece = float(by_task.get(task, {}).get("expected_calibration_error", 0.0))
        if ece > 0.10:
            calibration_warnings.append(f"{task} ECE={ece:.4f}")
    checks.append(
        Check(
            "WARN" if calibration_warnings else "PASS",
            "Calibration claim boundary",
            "; ".join(calibration_warnings) if calibration_warnings else "All checked ECE values <= 0.10",
            "Report confidence as model confidence, not calibrated clinical probability." if calibration_warnings else "",
        )
    )
    return checks, [by_task[task] for task in EVIDENCE_LABELS if task in by_task]


def check_vlm(rows: list[dict[str, str]], primary_rows: list[dict[str, str]]) -> list[Check]:
    direct_rows = [row for row in rows if row["family"] in {"VLM zero-shot", "VLM LoRA"}]
    hierarchical_rows = [row for row in rows if row["family"] == "VLM hierarchical diagnostic"]
    best_vlm = max((as_float(row, "accuracy") for row in rows), default=0.0)
    integrated_cnn_rows = [row for row in primary_rows if row["model"] in {"hierarchical", "eight_class"}]
    best_cnn = max((as_float(row, "accuracy") for row in integrated_cnn_rows), default=0.0)
    return [
        Check(
            "PASS" if direct_rows and hierarchical_rows else "FAIL",
            "Multimodal VLM comparison",
            f"best VLM={pct(best_vlm)}, best integrated CNN={pct(best_cnn)}",
            "Use VLMs as report/metadata helpers, not direct classifiers." if best_vlm < best_cnn else "Reassess VLM conclusion.",
        )
    ]


def result_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    ordered = sorted(rows, key=lambda row: list(PRIMARY_MODELS).index(row["model"]))
    return [
        [
            row["label"],
            f"{int(float(row['n'])):,}",
            pct(as_float(row, "accuracy")),
            pct(as_float(row, "macro_f1")),
            f"{pct(as_float(row, 'accuracy_ci95_low'))}-{pct(as_float(row, 'accuracy_ci95_high'))}",
        ]
        for row in ordered
    ]


def evidence_rows(rows: list[dict]) -> list[list[str]]:
    return [
        [
            EVIDENCE_LABELS.get(row["task"], row["task"]),
            f"{int(row['n']):,}",
            pct(float(row["accuracy"])),
            pct(float(row["expected_calibration_error"])),
            pct(float(row["multiclass_brier"])),
            pct(float(row.get("roc_auc_macro_ovr", 0.0))),
            pct(float(row.get("average_precision_macro", 0.0))),
        ]
        for row in rows
    ]


def write_status_doc(
    checks: list[Check],
    primary: list[dict[str, str]],
    sensitivity: list[dict[str, str]],
    evidence: list[dict],
) -> None:
    failures = [check for check in checks if check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN"]
    overall = "BLOCKED" if failures else "READY_WITH_LIMITATIONS"
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    lines = [
        "# Publication Submission Status",
        "",
        f"Generated by `scripts/check_publication_package.py` at `{generated}`.",
        "",
        f"Overall status: `{overall}`",
        "",
        (
            "The package is ready for internal manuscript drafting and advisor review. "
            "It is not ready for clinical-deployment claims because external validation and patient-level metadata are still absent."
        ),
        "",
        "## Primary Claim",
        "",
        (
            "The exact-deduplicated CNN suite is the primary result set. The single 8-class CNN is the top strict-test classifier, "
            "while the hierarchical CNN remains useful for interpretable routing and specialist analysis."
        ),
        "",
    ]
    lines.extend(md_table(["Model", "N", "Accuracy", "Macro F1", "95% CI"], result_rows(primary)))
    lines.extend(
        [
            "",
            "## Robustness Claim",
            "",
            (
                "The conservative dHash sensitivity run removes both exact SHA-256 and identical-dHash cross-split overlap. "
                "It supports the main result but should be presented as robustness evidence, not as the default checkpoint set."
            ),
            "",
        ]
    )
    lines.extend(md_table(["Model", "N", "Accuracy", "Macro F1", "95% CI"], result_rows(sensitivity)))
    lines.extend(
        [
            "",
            "## Probability Evidence",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Model", "N", "Accuracy", "ECE", "Brier", "ROC AUC", "Average Precision"],
            evidence_rows(evidence),
        )
    )
    lines.extend(
        [
            "",
            "## Validation Gate",
            "",
        ]
    )
    lines.extend(md_table(["Status", "Check", "Evidence", "Action"], [[c.status, c.item, c.evidence, c.action] for c in checks]))
    lines.extend(
        [
            "",
            "## Reviewer-Facing Evidence Map",
            "",
            "- Tables: `docs/PUBLICATION_RESULTS_TABLES.md`, `docs/publication_cnn_results.csv`, `docs/publication_audit_checks.csv`, `docs/publication_vlm_results.csv`.",
            "- Dataset provenance: `docs/DATASET_PROVENANCE.md`.",
            "- Full manuscript draft: `docs/MANUSCRIPT_FULL_DRAFT.md`.",
            "- Figures: `docs/figures/figure1_workflow.png` through `docs/figures/figure7_roc_pr_curves.png`.",
            "- Captions: `docs/FIGURE_CAPTIONS.md`.",
            "- Probability evidence: `training_logs/publication_evidence/`.",
            "- Leakage audits: `training_logs/publication_audit/`.",
            "- Grounded report path: `docs/GROUNDED_REPORTING.md` and `src/grounded_report.py`.",
            "",
            "## Regeneration Commands",
            "",
            "```bash",
            "python3 scripts/build_publication_tables.py",
            "python3 scripts/build_publication_figures.py",
            "python3 scripts/check_publication_package.py",
            "```",
            "",
            "Heavy retraining should remain on Kaggle or another GPU runner. Local runs should be limited to smoke tests and documentation checks.",
            "",
            "## Remaining Before Submission",
            "",
            "- Choose the exact target venue and reformat the manuscript to its required template.",
            "- Replace manuscript placeholders for authors, affiliations, funding, conflicts, and formal references.",
            "- Re-check dataset citations, license statements, and attribution text against the live Kaggle data cards.",
            "- State clearly that patient-level metadata were unavailable, so patient-level leakage cannot be excluded.",
            "- State clearly that external MRI validation is required before clinical or deployment claims.",
            "- Decide whether to include the deterministic report generator as an application contribution or keep it as a safety-oriented appendix.",
        ]
    )
    if warnings:
        lines.extend(
            [
                "",
                "## Active Warnings",
                "",
            ]
        )
        lines.extend(f"- {warning.item}: {warning.evidence}. {warning.action}".rstrip() for warning in warnings)
    if failures:
        lines.extend(
            [
                "",
                "## Blocking Failures",
                "",
            ]
        )
        lines.extend(f"- {failure.item}: {failure.evidence}. {failure.action}".rstrip() for failure in failures)

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    checks = check_required_paths()
    cnn_rows = read_csv("docs/publication_cnn_results.csv")
    audit_rows = read_csv("docs/publication_audit_checks.csv")
    vlm_rows = read_csv("docs/publication_vlm_results.csv")
    evidence_json = read_json("training_logs/publication_evidence/publication_evidence_summary.json")

    cnn_checks, primary, sensitivity = check_cnn_results(cnn_rows)
    checks.extend(cnn_checks)
    checks.extend(check_audits(audit_rows))
    evidence_checks, evidence = check_evidence(evidence_json)
    checks.extend(evidence_checks)
    checks.extend(check_vlm(vlm_rows, primary))

    write_status_doc(checks, primary, sensitivity, evidence)

    failures = [check for check in checks if check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN"]
    print(f"Wrote {rel(OUTPUT_PATH)}")
    print(f"Checks: {len(checks)} total, {len(failures)} fail, {len(warnings)} warn")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
