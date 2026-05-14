#!/usr/bin/env python3
"""Build final publication figure panels from committed artifacts."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

BG = "#f7f8fa"
PANEL = "#ffffff"
INK = "#172033"
MUTED = "#586174"
LINE = "#c8ced8"
BLUE = "#2f6f9f"
GREEN = "#2f8f70"
ORANGE = "#d07a27"
RED = "#b84a45"
TEAL = "#338a9a"
GRAY = "#8a93a3"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    max_width: int,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
    spacing: int = 8,
) -> int:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if text_size(draw, candidate, fnt)[0] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    x, y = xy
    line_height = text_size(draw, "Ag", fnt)[1] + spacing
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = PANEL,
    outline: str = LINE,
    radius: int = 20,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, fill: str = MUTED, width: int = 5) -> None:
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 18
    left = (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6))
    right = (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6))
    draw.polygon([end, left, right], fill=fill)


def make_canvas(width: int, height: int, title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), title, font=font(46, bold=True), fill=INK)
    if subtitle:
        draw.text((72, 108), subtitle, font=font(25), fill=MUTED)
    return image, draw


def save(image: Image.Image, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    image.save(path, dpi=(300, 300))
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    return (
        read_csv(DOCS_DIR / "publication_cnn_results.csv"),
        read_csv(DOCS_DIR / "publication_vlm_results.csv"),
        read_csv(DOCS_DIR / "publication_audit_checks.csv"),
    )


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    accent: str,
    title_size: int = 27,
    body_size: int = 22,
) -> None:
    rounded_panel(draw, box)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1, y1, x1 + 15, y2), radius=12, fill=accent)
    draw.text((x1 + 32, y1 + 24), title, font=font(title_size, bold=True), fill=INK)
    draw_wrapped(draw, body, (x1 + 32, y1 + 66), max_width=x2 - x1 - 64, fnt=font(body_size), fill=MUTED)


def figure1_workflow() -> Path:
    _, _, audit_rows = metric_rows()
    audit_by_name = {row["analysis"]: row for row in audit_rows}

    image, draw = make_canvas(
        2600,
        1550,
        "Figure 1. Dataset splitting and leakage-audit workflow",
        "Strict splits were created before augmentation; exact duplicate leakage was corrected before final claims.",
    )

    boxes = [
        (
            (90, 230, 520, 470),
            "MRI source datasets",
            "Brain tumor MRI: glioma, meningioma, notumor, pituitary. Dementia MRI: four dementia-stage classes.",
            BLUE,
        ),
        (
            (650, 230, 1080, 470),
            "Strict manifest",
            "51,023 image rows split into train, validation, and test before applying training-only augmentation.",
            TEAL,
        ),
        (
            (1210, 230, 1640, 470),
            "Initial audit",
            f"{audit_by_name['Initial leaky audit']['exact_hash_overlaps']} exact cross-split overlaps made the first result non-publishable.",
            RED,
        ),
        (
            (1770, 230, 2200, 470),
            "Exact-dedup retrain",
            f"{audit_by_name['Accepted exact-deduplicated']['exact_hash_overlaps']} exact overlaps. Accepted baseline checkpoints and metrics.",
            GREEN,
        ),
        (
            (1010, 750, 1440, 990),
            "dHash sensitivity",
            f"{audit_by_name['Conservative dHash sensitivity']['exact_hash_overlaps']} exact and {audit_by_name['Conservative dHash sensitivity']['perceptual_hash_overlaps']} dHash overlaps. Robustness evidence.",
            ORANGE,
        ),
    ]
    for box, title, body, accent in boxes:
        draw_box(draw, box, title, body, accent=accent)

    arrow(draw, (520, 350), (650, 350))
    arrow(draw, (1080, 350), (1210, 350))
    arrow(draw, (1640, 350), (1770, 350))
    arrow(draw, (1985, 470), (1440, 750), fill=ORANGE)

    draw_box(
        draw,
        (90, 1110, 820, 1370),
        "Training controls",
        "Train-only augmentation: flips, rotations, contrast jitter, random erasing, class weights, label smoothing, AdamW, and early stopping.",
        accent=BLUE,
        body_size=21,
    )
    draw_box(
        draw,
        (935, 1110, 1665, 1370),
        "Publication artifacts",
        "Checkpoints, confusion matrices, strict-test metrics, probability CSVs, calibration curves, ROC curves, and PR curves.",
        accent=GREEN,
        body_size=21,
    )
    draw_box(
        draw,
        (1780, 1110, 2510, 1370),
        "Claim boundary",
        "Strong internal MRI dataset performance. No clinical deployment claim without external validation and patient-level metadata.",
        accent=RED,
        body_size=21,
    )

    return save(image, "figure1_workflow.png")


def figure2_architecture() -> Path:
    image, draw = make_canvas(
        2600,
        1550,
        "Figure 2. CNN architecture comparison",
        "Hierarchical routing improves interpretability; the single 8-class head is the top strict-test performer.",
    )

    # Shared input
    draw_box(draw, (90, 680, 410, 880), "Input", "Brain MRI image", accent=GRAY)

    # Hierarchical path
    draw.text((560, 210), "Hierarchical CNN", font=font(36, bold=True), fill=INK)
    draw_box(draw, (520, 300, 920, 520), "Binary router", "ResNet50 tumor vs dementia. Strict-test accuracy: 1.0000.", accent=BLUE)
    draw_box(draw, (1050, 190, 1510, 410), "Tumor specialist", "EfficientNet-B3. Tumor subtype accuracy: 0.9792.", accent=ORANGE)
    draw_box(draw, (1050, 520, 1510, 740), "Dementia specialist", "MobileNetV3-Large. Dementia subtype accuracy: 0.9991.", accent=GREEN)
    draw_box(draw, (1630, 340, 2070, 610), "Routed 8-class output", "End-to-end hierarchical accuracy: 0.9963. Macro F1: 0.9891.", accent=TEAL)

    arrow(draw, (410, 780), (520, 410))
    arrow(draw, (920, 410), (1050, 300))
    arrow(draw, (920, 410), (1050, 630))
    arrow(draw, (1510, 300), (1630, 430))
    arrow(draw, (1510, 630), (1630, 520))

    # Single head path
    draw.text((560, 925), "Single-head CNN", font=font(36, bold=True), fill=INK)
    draw_box(draw, (520, 1010, 1030, 1240), "Single 8-class model", "EfficientNet-B3 trained directly over tumor and dementia classes.", accent=BLUE)
    draw_box(draw, (1160, 1010, 1660, 1240), "8-class output", "Strict-test accuracy: 0.9972. Macro F1: 0.9905.", accent=GREEN)
    arrow(draw, (410, 780), (520, 1125))
    arrow(draw, (1030, 1125), (1160, 1125))

    draw_box(
        draw,
        (1850, 940, 2470, 1260),
        "Publication comparison",
        "Report the single-head model as the highest-accuracy baseline, and keep the hierarchy for routed specialist analysis and safer user-facing reports.",
        accent=RED,
        body_size=22,
    )

    return save(image, "figure2_architecture.png")


def paste_image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    source: Path,
    box: tuple[int, int, int, int],
    title: str,
    label: str,
) -> None:
    x1, y1, x2, y2 = box
    rounded_panel(draw, box, radius=16, width=2)
    draw.text((x1 + 22, y1 + 18), label, font=font(28, bold=True), fill=INK)
    draw.text((x1 + 72, y1 + 20), title, font=font(25, bold=True), fill=INK)

    with Image.open(source).convert("RGB") as src:
        max_w = x2 - x1 - 44
        max_h = y2 - y1 - 78
        scale = min(max_w / src.width, max_h / src.height)
        resized = src.resize((int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS)
        px = x1 + (x2 - x1 - resized.width) // 2
        py = y1 + 64 + (max_h - resized.height) // 2
        canvas.paste(resized, (px, py))


def confusion_panel(name: str, title: str, sources: list[tuple[str, str, Path]]) -> Path:
    image, draw = make_canvas(2600, 1900, title)
    boxes = [
        (90, 180, 1260, 970),
        (1340, 180, 2510, 970),
        (90, 1040, 1260, 1830),
        (1340, 1040, 2510, 1830),
    ]
    for box, (label, subtitle, source) in zip(boxes, sources):
        paste_image_panel(image, draw, source, box, subtitle, label)
    return save(image, name)


def figure3_exact_confusion() -> Path:
    return confusion_panel(
        "figure3_exact_dedup_confusion.png",
        "Figure 3. Accepted exact-deduplicated strict-test confusion matrices",
        [
            ("A", "Tumor specialist", PROJECT_ROOT / "training_logs/experiments_dedup_regularized/tumor/20260512_021514/test/confusion_matrix.png"),
            ("B", "Dementia specialist", PROJECT_ROOT / "training_logs/experiments_dedup_regularized/dementia/20260512_022515/test/confusion_matrix.png"),
            ("C", "Hierarchical CNN", PROJECT_ROOT / "training_logs/experiments_dedup_regularized/hierarchical/test_evaluation/confusion_matrix.png"),
            ("D", "Single 8-class CNN", PROJECT_ROOT / "training_logs/experiments_dedup_regularized/eight_class/20260512_025559/test/confusion_matrix.png"),
        ],
    )


def figure4_dhash_confusion() -> Path:
    return confusion_panel(
        "figure4_dhash_sensitivity_confusion.png",
        "Figure 4. Conservative dHash sensitivity strict-test confusion matrices",
        [
            ("A", "Tumor specialist", PROJECT_ROOT / "training_logs/experiments_perceptual_regularized/tumor/20260513_162514/test/confusion_matrix.png"),
            ("B", "Dementia specialist", PROJECT_ROOT / "training_logs/experiments_perceptual_regularized/dementia/20260513_163515/test/confusion_matrix.png"),
            ("C", "Hierarchical CNN", PROJECT_ROOT / "training_logs/experiments_perceptual_regularized/hierarchical/test_evaluation/confusion_matrix.png"),
            ("D", "Single 8-class CNN", PROJECT_ROOT / "training_logs/experiments_perceptual_regularized/eight_class/20260513_170749/test/confusion_matrix.png"),
        ],
    )


def bar_chart(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, float, str]],
    *,
    box: tuple[int, int, int, int],
    x_max: float,
    title: str,
) -> None:
    x1, y1, x2, y2 = box
    rounded_panel(draw, box, radius=18, width=2)
    draw.text((x1 + 30, y1 + 26), title, font=font(31, bold=True), fill=INK)
    chart_left = x1 + 420
    chart_right = x2 - 90
    bar_h = 54
    gap = 35
    top = y1 + 105
    axis_y = top + len(rows) * (bar_h + gap) + 10
    draw.line((chart_left, axis_y, chart_right, axis_y), fill=LINE, width=3)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = chart_left + int((chart_right - chart_left) * tick / x_max)
        draw.line((tx, axis_y - 8, tx, axis_y + 8), fill=LINE, width=2)
        draw.text((tx - 24, axis_y + 16), f"{tick:.2f}", font=font(18), fill=MUTED)

    for idx, (label, value, color) in enumerate(rows):
        y = top + idx * (bar_h + gap)
        draw.text((x1 + 30, y + 12), label, font=font(24, bold=True), fill=INK)
        bar_w = int((chart_right - chart_left) * value / x_max)
        draw.rounded_rectangle((chart_left, y, chart_left + bar_w, y + bar_h), radius=12, fill=color)
        draw.text((chart_left + bar_w + 18, y + 11), f"{value:.4f}", font=font(24, bold=True), fill=INK)


def figure5_cnn_vlm_comparison() -> Path:
    image, draw = make_canvas(
        2400,
        1400,
        "Figure 5. CNN classifiers outperform direct multimodal VLM labeling",
        "VLMs are useful as experimental report/metadata helpers, but not as direct MRI classifiers in this study.",
    )
    rows = [
        ("Single 8-class CNN", 0.9972, GREEN),
        ("Hierarchical CNN", 0.9963, TEAL),
        ("Qwen2.5-VL-7B domain routing", 0.8750, BLUE),
        ("Qwen2.5-VL-3B LoRA direct", 0.3125, ORANGE),
        ("Qwen2.5-VL-7B routed 8-class", 0.2250, RED),
        ("Qwen2.5-VL-7B zero-shot", 0.1750, GRAY),
    ]
    bar_chart(draw, rows, box=(120, 230, 2280, 940), x_max=1.0, title="Strict-test or benchmark accuracy")
    draw_box(
        draw,
        (120, 1030, 1110, 1260),
        "Conclusion",
        "Keep CNNs as the diagnostic image classifiers. Present VLM experiments as negative direct-classification results and use grounded reports for user-facing text.",
        accent=GREEN,
        body_size=23,
    )
    draw_box(
        draw,
        (1240, 1030, 2280, 1260),
        "Important caveat",
        "CNN test sets are much larger than the VLM samples, so this figure is an architectural comparison, not a statistical equivalence test.",
        accent=ORANGE,
        body_size=23,
    )
    return save(image, "figure5_cnn_vlm_comparison.png")


def evidence_panel(name: str, title: str, suffix_left: str, suffix_right: str, left_title: str, right_title: str) -> Path:
    image, draw = make_canvas(2700, 3100, title)
    row_labels = [
        ("Tumor specialist", "tumor"),
        ("Dementia specialist", "dementia"),
        ("Single 8-class CNN", "eight_class"),
        ("Hierarchical CNN", "hierarchical"),
    ]
    draw.text((560, 165), left_title, font=font(32, bold=True), fill=INK)
    draw.text((1660, 165), right_title, font=font(32, bold=True), fill=INK)
    y = 230
    for idx, (label, folder) in enumerate(row_labels):
        draw.text((90, y + 270), f"{chr(65 + idx)}. {label}", font=font(30, bold=True), fill=INK)
        paste_image_panel(
            image,
            draw,
            PROJECT_ROOT / f"training_logs/publication_evidence/{folder}/{suffix_left}",
            (410, y, 1440, y + 650),
            left_title,
            "",
        )
        paste_image_panel(
            image,
            draw,
            PROJECT_ROOT / f"training_logs/publication_evidence/{folder}/{suffix_right}",
            (1530, y, 2560, y + 650),
            right_title,
            "",
        )
        y += 700
    return save(image, name)


def figure6_calibration_confidence() -> Path:
    return evidence_panel(
        "figure6_calibration_confidence.png",
        "Figure 6. Calibration and confidence evidence for accepted CNN checkpoints",
        "calibration.png",
        "confidence_histogram.png",
        "Calibration",
        "Confidence histogram",
    )


def figure7_roc_pr() -> Path:
    return evidence_panel(
        "figure7_roc_pr_curves.png",
        "Figure 7. ROC and precision-recall evidence for accepted CNN checkpoints",
        "roc_curves.png",
        "precision_recall_curves.png",
        "ROC curves",
        "Precision-recall curves",
    )


CAPTIONS = {
    "figure1_workflow.png": (
        "Figure 1. Dataset And Audit Workflow",
        "Figure 1. Dataset splitting and leakage-audit workflow. Brain tumor and dementia MRI images were split before augmentation, audited for exact and perceptual cross-split overlap, then retrained after exact duplicate grouping. The dHash-grouped run is reported as a conservative sensitivity analysis.",
    ),
    "figure2_architecture.png": (
        "Figure 2. Model Architecture",
        "Figure 2. CNN architecture comparison. The hierarchical model routes images through a binary tumor-versus-dementia classifier and domain-specific specialists; the single-head baseline trains one EfficientNet-B3 classifier across all eight classes.",
    ),
    "figure3_exact_dedup_confusion.png": (
        "Figure 3. Exact-Deduplicated Confusion Matrices",
        "Figure 3. Accepted exact-deduplicated strict-test confusion matrices for tumor specialist, dementia specialist, hierarchical CNN, and single 8-class CNN checkpoints.",
    ),
    "figure4_dhash_sensitivity_confusion.png": (
        "Figure 4. dHash Sensitivity Confusion Matrices",
        "Figure 4. Conservative dHash sensitivity confusion matrices after grouping exact SHA-256 duplicates and identical dHash fingerprints into a single split.",
    ),
    "figure5_cnn_vlm_comparison.png": (
        "Figure 5. CNN And VLM Comparison",
        "Figure 5. CNN versus multimodal VLM comparison. CNN classifiers substantially outperform zero-shot, hierarchical, and LoRA-adapted VLM approaches for direct MRI image labeling in these experiments.",
    ),
    "figure6_calibration_confidence.png": (
        "Figure 6. Calibration And Confidence",
        "Figure 6. Calibration and confidence evidence. Domain specialists show lower expected calibration error than the 8-class and hierarchical models, so softmax scores should be reported as model confidence rather than calibrated clinical probability.",
    ),
    "figure7_roc_pr_curves.png": (
        "Figure 7. ROC And Precision-Recall Curves",
        "Figure 7. One-vs-rest ROC and precision-recall evidence for accepted CNN checkpoints, generated from full strict-test probability outputs.",
    ),
}


def write_caption_doc(paths: list[Path]) -> Path:
    lines = [
        "# Figure Captions",
        "",
        "Generated by `scripts/build_publication_figures.py` from committed experiment artifacts.",
        "",
    ]
    for path in paths:
        rel = path.relative_to(PROJECT_ROOT)
        heading, caption = CAPTIONS[path.name]
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(f"Artifact: `{rel}`")
        lines.append("")
        lines.append(caption)
        lines.append("")
    output = DOCS_DIR / "FIGURE_CAPTIONS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        figure1_workflow(),
        figure2_architecture(),
        figure3_exact_confusion(),
        figure4_dhash_confusion(),
        figure5_cnn_vlm_comparison(),
        figure6_calibration_confidence(),
        figure7_roc_pr(),
    ]
    caption_doc = write_caption_doc(paths)
    for path in paths:
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {caption_doc.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
