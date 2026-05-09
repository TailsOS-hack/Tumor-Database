#!/usr/bin/env python3
"""Collect strict-test metrics into publication-ready summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_logs" / "experiments"
TASKS = ["binary", "tumor", "dementia", "hierarchical", "eight_class"]


def latest_metrics(output_dir: Path, task: str) -> Path | None:
    if task == "hierarchical":
        path = output_dir / "hierarchical" / "test_evaluation" / "metrics.json"
        return path if path.exists() else None

    candidates = sorted((output_dir / task).glob("*/test/metrics.json"))
    return candidates[-1] if candidates else None


def metric_value(report: dict, key: str, metric: str) -> float | None:
    value = report.get(key, {})
    if isinstance(value, dict):
        raw = value.get(metric)
        return float(raw) if raw is not None else None
    return None


def summarize(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASKS:
        path = latest_metrics(output_dir, task)
        if not path:
            rows.append({"model": task, "status": "missing", "metrics_path": ""})
            continue

        metrics = json.loads(path.read_text(encoding="utf-8"))
        report = metrics.get("classification_report", {})
        rows.append(
            {
                "model": task,
                "status": "complete",
                "accuracy": metrics.get("accuracy", report.get("accuracy")),
                "macro_f1": metric_value(report, "macro avg", "f1-score"),
                "weighted_f1": metric_value(report, "weighted avg", "f1-score"),
                "metrics_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def write_markdown(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_path = output_dir / "publication_summary.md"
    lines = [
        "# Publication Summary",
        "",
        "| Model | Status | Accuracy | Macro F1 | Weighted F1 | Metrics Path |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {status} | {accuracy} | {macro_f1} | {weighted_f1} | `{metrics_path}` |".format(
                model=row["model"],
                status=row["status"],
                accuracy=format_float(row.get("accuracy")),
                macro_f1=format_float(row.get("macro_f1")),
                weighted_f1=format_float(row.get("weighted_f1")),
                metrics_path=row.get("metrics_path", ""),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def format_float(value: object) -> str:
    if value is None or value == "":
        return "TBD"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    rows = summarize(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "publication_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown_path = write_markdown(rows, output_dir)
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
