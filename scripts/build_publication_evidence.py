#!/usr/bin/env python3
"""Build publication evidence figures from trained checkpoints.

This script intentionally evaluates existing checkpoints; it does not retrain.
It produces probability CSVs, calibration summaries, ROC/PR figures, confidence
histograms, and a compact Markdown summary for manuscript figure assembly.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

try:
    from src import experiment_pipeline as ep
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src import experiment_pipeline as ep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_logs" / "publication_evidence"

TASKS = ["binary", "tumor", "dementia", "eight_class"]
HIERARCHICAL_TASK = "hierarchical"
np = None


def get_numpy():
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def print_step(message: str) -> None:
    print(f"[publication-evidence] {message}", flush=True)


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_hot(y_true: np.ndarray, class_count: int) -> np.ndarray:
    np = get_numpy()
    out = np.zeros((len(y_true), class_count), dtype=float)
    for index, value in enumerate(y_true):
        out[index, int(value)] = 1.0
    return out


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clean probability rows before sklearn metrics and plots.

    Some GPU kernels can produce non-finite values when inference is run through
    mixed precision on a checkpoint that was trained in full precision. The
    publication evidence job is evaluation-only, so we prefer stable fp32
    probabilities and a defensive normalization pass over speed.
    """

    np = get_numpy()
    clean = np.asarray(probabilities, dtype=float)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, None)
    row_sums = clean.sum(axis=1, keepdims=True)
    empty_rows = row_sums.squeeze(1) <= 0
    if empty_rows.any():
        clean[empty_rows, :] = 1.0 / clean.shape[1]
        row_sums = clean.sum(axis=1, keepdims=True)
    return clean / row_sums


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    np = get_numpy()
    rows: list[dict[str, Any]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    correctness = (y_true == y_pred).astype(float)

    for bin_index in range(bins):
        low = edges[bin_index]
        high = edges[bin_index + 1]
        if bin_index == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        if count:
            bin_acc = float(correctness[mask].mean())
            bin_conf = float(confidence[mask].mean())
            gap = abs(bin_acc - bin_conf)
            ece += gap * count / max(1, len(y_true))
        else:
            bin_acc = None
            bin_conf = None
            gap = None
        rows.append(
            {
                "bin": bin_index,
                "low": float(low),
                "high": float(high),
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": gap,
            }
        )
    return ece, rows


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    *,
    bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    np = get_numpy()
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    probabilities = normalize_probabilities(probabilities)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    class_count = len(class_names)
    y_onehot = one_hot(y_true, class_count)
    ece, calibration_rows = expected_calibration_error(y_true, y_pred, confidence, bins=bins)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "correct": int((y_true == y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mean_confidence": float(confidence.mean()) if len(confidence) else None,
        "median_confidence": float(np.median(confidence)) if len(confidence) else None,
        "expected_calibration_error": float(ece),
        "multiclass_brier": float(np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "class_names": class_names,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(class_count)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        metrics["log_loss"] = float(log_loss(y_true, probabilities, labels=list(range(class_count))))
    except Exception as exc:
        metrics["log_loss_error"] = str(exc)

    try:
        if class_count == 2:
            metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            metrics["average_precision_macro"] = float(average_precision_score(y_true, probabilities[:, 1]))
        else:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_true, probabilities, labels=list(range(class_count)), multi_class="ovr", average="macro")
            )
            metrics["average_precision_macro"] = float(average_precision_score(y_onehot, probabilities, average="macro"))
    except Exception as exc:
        metrics["auc_error"] = str(exc)

    return metrics, calibration_rows


def prediction_rows(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    image_paths: list[str],
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    probabilities = normalize_probabilities(probabilities)
    y_pred = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index, (true, pred) in enumerate(zip(y_true, y_pred)):
        row: dict[str, Any] = {
            "index": index,
            "image_path": image_paths[index],
            "true_index": int(true),
            "pred_index": int(pred),
            "true_label": class_names[int(true)],
            "pred_label": class_names[int(pred)],
            "confidence": float(probabilities[index, int(pred)]),
            "correct": int(true == pred),
        }
        if extra:
            row.update(extra[index])
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probabilities[index, class_index])
        rows.append(row)
    return rows


def plot_calibration(path: Path, calibration_rows: list[dict[str, Any]], title: str) -> None:
    plt = ensure_matplotlib()
    centers = [(row["low"] + row["high"]) / 2 for row in calibration_rows]
    acc = [row["accuracy"] if row["accuracy"] is not None else math.nan for row in calibration_rows]
    conf = [row["confidence"] if row["confidence"] is not None else math.nan for row in calibration_rows]
    counts = [row["count"] for row in calibration_rows]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#777", label="Perfect calibration")
    ax1.plot(centers, acc, marker="o", label="Accuracy")
    ax1.plot(centers, conf, marker="s", label="Confidence")
    ax1.set_xlabel("Confidence bin")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.02)
    ax1.set_title(title)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=0.08, alpha=0.18, color="#4c78a8", label="Count")
    ax2.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confidence_histogram(path: Path, y_true: np.ndarray, probabilities: np.ndarray, title: str) -> None:
    plt = ensure_matplotlib()
    probabilities = normalize_probabilities(probabilities)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = confidence[y_true == y_pred]
    incorrect = confidence[y_true != y_pred]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(correct, bins=20, alpha=0.7, label=f"Correct (n={len(correct)})")
    if len(incorrect):
        ax.hist(incorrect, bins=20, alpha=0.7, label=f"Incorrect (n={len(incorrect)})")
    ax.set_xlabel("Predicted-class confidence")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curves(path: Path, y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], title: str) -> None:
    from sklearn.metrics import auc, roc_curve

    plt = ensure_matplotlib()
    probabilities = normalize_probabilities(probabilities)
    y_onehot = one_hot(y_true, len(class_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for class_index, class_name in enumerate(class_names):
        if y_onehot[:, class_index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_onehot[:, class_index], probabilities[:, class_index])
        ax.plot(fpr, tpr, label=f"{class_name} AUC={auc(fpr, tpr):.3f}")
        plotted += 1
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_curves(path: Path, y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], title: str) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt = ensure_matplotlib()
    probabilities = normalize_probabilities(probabilities)
    y_onehot = one_hot(y_true, len(class_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for class_index, class_name in enumerate(class_names):
        if y_onehot[:, class_index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_onehot[:, class_index], probabilities[:, class_index])
        ap = average_precision_score(y_onehot[:, class_index], probabilities[:, class_index])
        ax.plot(recall, precision, label=f"{class_name} AP={ap:.3f}")
        plotted += 1
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_task(args: argparse.Namespace, task: str, output_dir: Path) -> dict[str, Any]:
    np = get_numpy()
    import torch
    from torch.utils.data import DataLoader

    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    run_device = device()
    checkpoint_path = Path(args.checkpoint_overrides.get(task, ep.TASK_DEFAULT_MODEL_PATH[task]))
    model, checkpoint = ep.load_checkpoint_model(checkpoint_path, run_device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", args.image_size))
    max_samples = args.max_samples_per_class
    samples = ep.select_samples(Path(args.manifest), split=args.split, task=task, max_samples_per_class=max_samples)

    dataset = ep.ManifestImageDataset(samples, ep.build_transforms(train=False, image_size=image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=run_device.type == "cuda",
    )
    y_true: list[int] = []
    probs: list[np.ndarray] = []
    image_paths = [str(path.relative_to(PROJECT_ROOT)) for path, _ in samples]

    print_step(f"Evaluating {task}: {len(samples)} samples on {run_device}")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(run_device)
            logits = model(inputs)
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.extend(batch_probs)
            y_true.extend(labels.cpu().numpy().astype(int).tolist())

    y_true_np = np.array(y_true, dtype=int)
    probs_np = np.array(probs, dtype=float)
    metrics, calibration_rows = metrics_from_probabilities(y_true_np, probs_np, class_names, bins=args.calibration_bins)
    rows = prediction_rows(y_true_np, probs_np, class_names, image_paths)

    write_csv(task_dir / "probabilities.csv", rows)
    write_csv(task_dir / "calibration_bins.csv", calibration_rows)
    write_json(task_dir / "metrics.json", metrics)
    plot_calibration(task_dir / "calibration.png", calibration_rows, f"{task} reliability")
    plot_confidence_histogram(task_dir / "confidence_histogram.png", y_true_np, probs_np, f"{task} confidence")
    plot_roc_curves(task_dir / "roc_curves.png", y_true_np, probs_np, class_names, f"{task} ROC")
    plot_pr_curves(task_dir / "precision_recall_curves.png", y_true_np, probs_np, class_names, f"{task} precision-recall")

    metrics["task"] = task
    metrics["checkpoint_path"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
    metrics["probabilities_path"] = str((task_dir / "probabilities.csv").relative_to(output_dir))
    return metrics


def evaluate_hierarchical(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    np = get_numpy()
    import torch
    from PIL import Image

    task_dir = output_dir / HIERARCHICAL_TASK
    task_dir.mkdir(parents=True, exist_ok=True)
    run_device = device()

    binary_model, binary_ckpt = ep.load_checkpoint_model(ep.TASK_DEFAULT_MODEL_PATH["binary"], run_device)
    tumor_model, tumor_ckpt = ep.load_checkpoint_model(ep.TASK_DEFAULT_MODEL_PATH["tumor"], run_device)
    dementia_model, dementia_ckpt = ep.load_checkpoint_model(ep.TASK_DEFAULT_MODEL_PATH["dementia"], run_device)

    binary_tf = ep.build_transforms(train=False, image_size=int(binary_ckpt.get("image_size", args.image_size)))
    tumor_tf = ep.build_transforms(train=False, image_size=int(tumor_ckpt.get("image_size", args.image_size)))
    dementia_tf = ep.build_transforms(train=False, image_size=int(dementia_ckpt.get("image_size", args.image_size)))

    rows = [row for row in ep.read_manifest(Path(args.manifest)) if row["split"] == args.split]
    if args.max_samples_per_class is not None:
        buckets: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            buckets.setdefault(row["eight_class"], []).append(row)
        rows = []
        for class_name in sorted(buckets):
            rows.extend(sorted(buckets[class_name], key=lambda item: item["path"])[: args.max_samples_per_class])

    y_true: list[int] = []
    probs: list[np.ndarray] = []
    image_paths: list[str] = []
    extra: list[dict[str, Any]] = []

    print_step(f"Evaluating hierarchical: {len(rows)} samples on {run_device}")
    with torch.no_grad():
        for row in rows:
            image = Image.open(PROJECT_ROOT / row["path"]).convert("RGB")
            binary_input = binary_tf(image).unsqueeze(0).to(run_device)
            binary_logits = binary_model(binary_input)
            binary_probs = torch.softmax(binary_logits, dim=1).squeeze(0)
            domain_idx = int(torch.argmax(binary_probs).item())
            predicted_domain = binary_ckpt["class_names"][domain_idx]

            tumor_input = tumor_tf(image).unsqueeze(0).to(run_device)
            dementia_input = dementia_tf(image).unsqueeze(0).to(run_device)
            tumor_logits = tumor_model(tumor_input)
            dementia_logits = dementia_model(dementia_input)

            tumor_probs = torch.softmax(tumor_logits, dim=1).squeeze(0)
            dementia_probs = torch.softmax(dementia_logits, dim=1).squeeze(0)
            full_probs = torch.cat(
                [
                    binary_probs[ep.DOMAIN_CLASSES.index("tumor")] * tumor_probs,
                    binary_probs[ep.DOMAIN_CLASSES.index("dementia")] * dementia_probs,
                ],
                dim=0,
            )

            if predicted_domain == "tumor":
                pred_subtype_idx = int(torch.argmax(tumor_probs).item())
                pred_idx = pred_subtype_idx
                pred_subtype = tumor_ckpt["class_names"][pred_subtype_idx]
            else:
                pred_subtype_idx = int(torch.argmax(dementia_probs).item())
                pred_idx = len(ep.TUMOR_CLASSES) + pred_subtype_idx
                pred_subtype = dementia_ckpt["class_names"][pred_subtype_idx]

            y_true.append(ep.EIGHT_CLASS_NAMES.index(row["eight_class"]))
            probs.append(full_probs.detach().cpu().numpy())
            image_paths.append(row["path"])
            extra.append(
                {
                    "router_pred_domain": predicted_domain,
                    "router_confidence": float(binary_probs[domain_idx].item()),
                    "specialist_pred_subtype": pred_subtype,
                    "routed_pred_index": pred_idx,
                    "routed_pred_label": ep.EIGHT_CLASS_NAMES[pred_idx],
                }
            )

    y_true_np = np.array(y_true, dtype=int)
    probs_np = np.array(probs, dtype=float)
    metrics, calibration_rows = metrics_from_probabilities(y_true_np, probs_np, ep.EIGHT_CLASS_NAMES, bins=args.calibration_bins)
    rows_out = prediction_rows(y_true_np, probs_np, ep.EIGHT_CLASS_NAMES, image_paths, extra=extra)

    # Keep routed predictions as the official hierarchical accuracy; the probability
    # vector is used for calibration/ROC summaries.
    routed_pred = np.array([row["routed_pred_index"] for row in rows_out], dtype=int)
    metrics["routed_accuracy"] = float((y_true_np == routed_pred).mean())
    metrics["routed_correct"] = int((y_true_np == routed_pred).sum())

    write_csv(task_dir / "probabilities.csv", rows_out)
    write_csv(task_dir / "calibration_bins.csv", calibration_rows)
    write_json(task_dir / "metrics.json", metrics)
    plot_calibration(task_dir / "calibration.png", calibration_rows, "hierarchical reliability")
    plot_confidence_histogram(task_dir / "confidence_histogram.png", y_true_np, probs_np, "hierarchical confidence")
    plot_roc_curves(task_dir / "roc_curves.png", y_true_np, probs_np, ep.EIGHT_CLASS_NAMES, "hierarchical ROC")
    plot_pr_curves(task_dir / "precision_recall_curves.png", y_true_np, probs_np, ep.EIGHT_CLASS_NAMES, "hierarchical precision-recall")

    metrics["task"] = HIERARCHICAL_TASK
    metrics["checkpoint_path"] = "models/binary_router.pt + models/brain_tumor_classifier.pt + models/alzheimers_classifier.pt"
    metrics["probabilities_path"] = str((task_dir / "probabilities.csv").relative_to(output_dir))
    return metrics


def markdown_summary(metrics_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Publication Evidence Summary",
        "",
        "This bundle was generated from existing checkpoints. It does not retrain models.",
        "",
        "| Model | N | Accuracy | Routed Acc. | Macro F1 | ECE | Brier | ROC AUC | AP | Mean Confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for metrics in metrics_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(metrics["task"]),
                    str(metrics.get("n", "NA")),
                    fmt(metrics.get("accuracy")),
                    fmt(metrics.get("routed_accuracy")),
                    fmt(metrics.get("macro_f1")),
                    fmt(metrics.get("expected_calibration_error")),
                    fmt(metrics.get("multiclass_brier")),
                    fmt(metrics.get("roc_auc_macro_ovr")),
                    fmt(metrics.get("average_precision_macro")),
                    fmt(metrics.get("mean_confidence")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "Each model folder contains:",
            "",
            "- `probabilities.csv`",
            "- `calibration_bins.csv`",
            "- `metrics.json`",
            "- `calibration.png`",
            "- `confidence_histogram.png`",
            "- `roc_curves.png`",
            "- `precision_recall_curves.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def parse_checkpoint_override(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--checkpoint must be TASK=PATH")
        task, path = value.split("=", 1)
        if task not in TASKS:
            raise argparse.ArgumentTypeError(f"Unsupported task override: {task}")
        overrides[task] = path
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ep.DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--skip-hierarchical", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[], help="Optional TASK=PATH checkpoint override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint_overrides = parse_checkpoint_override(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for task in TASKS:
        metrics_rows.append(evaluate_task(args, task, output_dir))
    if not args.skip_hierarchical:
        metrics_rows.append(evaluate_hierarchical(args, output_dir))

    write_json(output_dir / "publication_evidence_summary.json", metrics_rows)
    (output_dir / "PUBLICATION_EVIDENCE_SUMMARY.md").write_text(markdown_summary(metrics_rows), encoding="utf-8")
    print_step(f"Wrote evidence bundle to {output_dir}")


if __name__ == "__main__":
    main()
