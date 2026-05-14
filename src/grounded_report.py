"""Grounded, non-hallucinating report generation for classifier outputs.

The report builder in this module is deliberately deterministic. It does not
ask an LLM to interpret an MRI image, estimate lesion measurements, invent
anatomic locations, or produce radiology findings that were not supplied as
structured evidence.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_VERSION = "grounded_mri_report_v1"

MODEL_CONTEXT = {
    "tumor": {
        "model": "Brain tumor specialist",
        "accepted_accuracy": 0.9792,
        "accepted_test_n": 1445,
        "accepted_ci95": "0.9705-0.9854",
        "sensitivity_accuracy": 0.9539,
        "sensitivity_test_n": 2019,
        "sensitivity_ci95": "0.9439-0.9623",
    },
    "dementia": {
        "model": "Dementia specialist",
        "accepted_accuracy": 0.9991,
        "accepted_test_n": 8806,
        "accepted_ci95": "0.9982-0.9995",
        "sensitivity_accuracy": 0.9975,
        "sensitivity_test_n": 8881,
        "sensitivity_ci95": "0.9963-0.9984",
    },
    "normal_or_no_tumor": {
        "model": "Domain-specific specialist",
        "accepted_accuracy": None,
        "accepted_test_n": None,
        "accepted_ci95": None,
        "sensitivity_accuracy": None,
        "sensitivity_test_n": None,
        "sensitivity_ci95": None,
    },
}

TUMOR_LABELS = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary tumor",
    "notumor": "no tumor class",
}

DEMENTIA_LABELS = {
    "MildDemented": "mild dementia class",
    "ModerateDemented": "moderate dementia class",
    "NonDemented": "non-demented class",
    "VeryMildDemented": "very mild dementia class",
}

NORMAL_LABELS = {"Normal", "normal", "notumor", "NonDemented", "tumor_notumor", "dementia_NonDemented"}

UNSUPPORTED_FINDINGS = [
    "lesion size",
    "tumor location",
    "contrast enhancement",
    "edema",
    "midline shift",
    "mass effect",
    "ventricular caliber",
    "hippocampal volume",
    "cortical atrophy severity",
    "vascular abnormality",
]


def _clean(value: Any, default: str = "Not provided") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _pct(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, (int, float)):
        if 0 <= float(value) <= 1:
            return f"{float(value) * 100:.1f}%"
        return f"{float(value):.1f}%"
    text = _clean(value)
    if text.endswith("%"):
        return text
    try:
        numeric = float(text)
    except ValueError:
        return text
    if 0 <= numeric <= 1:
        return f"{numeric * 100:.1f}%"
    return f"{numeric:.1f}%"


def normalize_prediction(prediction: str) -> dict[str, str]:
    raw = _clean(prediction, "unknown")
    label = raw
    domain = "unknown"

    if raw.startswith("tumor_"):
        domain = "tumor"
        label = raw.removeprefix("tumor_")
    elif raw.startswith("dementia_"):
        domain = "dementia"
        label = raw.removeprefix("dementia_")
    elif raw in TUMOR_LABELS:
        domain = "normal_or_no_tumor" if raw == "notumor" else "tumor"
    elif raw in DEMENTIA_LABELS:
        domain = "normal_or_no_tumor" if raw == "NonDemented" else "dementia"
    elif raw in NORMAL_LABELS:
        domain = "normal_or_no_tumor"

    if domain == "tumor":
        display = TUMOR_LABELS.get(label, label)
    elif domain == "dementia":
        display = DEMENTIA_LABELS.get(label, label)
    elif domain == "normal_or_no_tumor":
        display = "normal/no-tumor class"
    else:
        display = raw

    return {
        "raw_label": raw,
        "domain": domain,
        "subtype": label,
        "display_label": display,
    }


def _clinical_sentence(normalized: dict[str, str]) -> str:
    domain = normalized["domain"]
    display = normalized["display_label"]
    if domain == "tumor":
        return f"Classifier output is most consistent with the {display} class."
    if domain == "dementia":
        return f"Classifier output is most consistent with the {display}."
    if domain == "normal_or_no_tumor":
        return "Classifier output is most consistent with a normal/no-tumor class."
    return f"Classifier output is {normalized['raw_label']}."


def _domain_caution(normalized: dict[str, str]) -> str:
    domain = normalized["domain"]
    if domain == "tumor":
        return (
            "The classifier does not provide lesion measurements, anatomic location, enhancement pattern, "
            "edema, or mass-effect assessment."
        )
    if domain == "dementia":
        return (
            "The classifier does not measure hippocampal volume, cortical atrophy, ventricular size, "
            "or exclude non-Alzheimer causes of cognitive symptoms."
        )
    if domain == "normal_or_no_tumor":
        return (
            "The classifier output should not be treated as proof of a normal clinical MRI; radiologist "
            "review of the source images is still required."
        )
    return "The classifier output was not mapped to a supported report domain."


def build_grounded_report(
    *,
    prediction: str,
    confidence: Any = None,
    patient_info: dict[str, Any] | None = None,
    exam_details: dict[str, Any] | None = None,
    report_date: str | None = None,
    model_used: str | None = None,
    source_image: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic report from structured classifier evidence."""

    normalized = normalize_prediction(prediction)
    patient_info = patient_info or {}
    exam_details = exam_details or {}
    domain = normalized["domain"]
    context = MODEL_CONTEXT.get(domain, MODEL_CONTEXT["normal_or_no_tumor"])
    confidence_text = _pct(confidence)
    clinical_sentence = _clinical_sentence(normalized)
    caution = _domain_caution(normalized)

    report = {
        "report_version": REPORT_VERSION,
        "report_header": {
            "title": "AI Draft MRI Brain Report",
            "status": "Draft - requires clinician review",
            "date": report_date or datetime.now().strftime("%B %d, %Y"),
        },
        "patient_information": {
            "name": _clean(patient_info.get("name")),
            "patient_id": _clean(patient_info.get("patient_id")),
            "dob": _clean(patient_info.get("dob")),
        },
        "clinical_information": {
            "reason_for_exam": _clean(exam_details.get("reason")),
            "clinical_history": _clean(exam_details.get("history")),
            "comparison": _clean(exam_details.get("comparison"), "None"),
        },
        "procedure_details": {
            "examination": "MRI brain image classification",
            "technique": _clean(exam_details.get("technique"), "MRI brain protocol not specified"),
            "contrast": _clean(exam_details.get("contrast"), "Not specified"),
        },
        "classifier_evidence": {
            "prediction": normalized["raw_label"],
            "normalized_domain": normalized["domain"],
            "normalized_subtype": normalized["subtype"],
            "display_label": normalized["display_label"],
            "confidence": confidence_text,
            "model_used": _clean(model_used),
            "source_image": _clean(source_image),
            "model_context": context,
        },
        "findings": {
            "classifier_finding": clinical_sentence,
            "scope_limit": caution,
            "cerebral_parenchyma": "Not independently assessed by this report maker beyond classifier output.",
            "extra_axial_spaces": "No statement generated; not assessed by classifier evidence.",
            "ventricles": "No statement generated; not assessed by classifier evidence.",
            "mass_effect": "No statement generated; not assessed by classifier evidence.",
            "vascular_structures": "No statement generated; not assessed by classifier evidence.",
            "bones_and_soft_tissues": "No statement generated; not assessed by classifier evidence.",
            "paranasal_sinuses_mastoids": "No statement generated; not assessed by classifier evidence.",
            "orbits": "No statement generated; not assessed by classifier evidence.",
        },
        "impression": [
            f"AI classifier result: {clinical_sentence}",
            f"Classifier confidence: {confidence_text}.",
            caution,
            "This draft is decision-support text only and must be reviewed with the original MRI images by a qualified clinician.",
        ],
        "safety": {
            "generation_mode": "deterministic_template_no_llm_image_interpretation",
            "unsupported_findings_not_generated": UNSUPPORTED_FINDINGS,
            "allowed_evidence_sources": [
                "classifier prediction label",
                "classifier confidence",
                "model checkpoint/evaluation context",
                "user-entered patient and exam details",
            ],
        },
    }
    return report


def render_report_markdown(report: dict[str, Any]) -> str:
    header = report["report_header"]
    patient = report["patient_information"]
    clinical = report["clinical_information"]
    procedure = report["procedure_details"]
    evidence = report["classifier_evidence"]
    context = evidence["model_context"]
    findings = report["findings"]
    accepted_accuracy = context.get("accepted_accuracy")
    sensitivity_accuracy = context.get("sensitivity_accuracy")
    accepted_text = "Not available" if accepted_accuracy is None else f"{accepted_accuracy:.4f}"
    sensitivity_text = "Not available" if sensitivity_accuracy is None else f"{sensitivity_accuracy:.4f}"

    lines = [
        f"# {header['title']}",
        "",
        f"**Status:** {header['status']}",
        f"**Date:** {header['date']}",
        "",
        "## Patient Information",
        "",
        f"- Name: {patient['name']}",
        f"- DOB: {patient['dob']}",
        f"- Patient ID: {patient['patient_id']}",
        "",
        "## Clinical Information",
        "",
        f"- Reason for exam: {clinical['reason_for_exam']}",
        f"- Clinical history: {clinical['clinical_history']}",
        f"- Comparison: {clinical['comparison']}",
        "",
        "## Procedure Details",
        "",
        f"- Examination: {procedure['examination']}",
        f"- Technique: {procedure['technique']}",
        f"- Contrast: {procedure['contrast']}",
        "",
        "## Classifier Evidence",
        "",
        f"- Prediction: {evidence['display_label']} (`{evidence['prediction']}`)",
        f"- Domain: {evidence['normalized_domain']}",
        f"- Confidence: {evidence['confidence']}",
        f"- Model used: {evidence['model_used']}",
        f"- Internal exact-deduplicated test accuracy: {accepted_text}",
        f"- Conservative dHash sensitivity accuracy: {sensitivity_text}",
        "",
        "## Findings",
        "",
        f"- Classifier finding: {findings['classifier_finding']}",
        f"- Scope limit: {findings['scope_limit']}",
        f"- Cerebral parenchyma: {findings['cerebral_parenchyma']}",
        f"- Extra-axial spaces: {findings['extra_axial_spaces']}",
        f"- Ventricles: {findings['ventricles']}",
        f"- Mass effect: {findings['mass_effect']}",
        "",
        "## Impression",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(report["impression"], start=1))
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- Generation mode: {report['safety']['generation_mode']}",
            "- Unsupported findings are intentionally not generated unless supplied by a clinician.",
        ]
    )
    return "\n".join(lines)


def render_report_html(report: dict[str, Any]) -> str:
    markdown_text = render_report_markdown(report)
    body_lines = []
    in_list = False
    in_ordered = False

    def close_lists() -> None:
        nonlocal in_list, in_ordered
        if in_list:
            body_lines.append("</ul>")
            in_list = False
        if in_ordered:
            body_lines.append("</ol>")
            in_ordered = False

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            close_lists()
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            close_lists()
            body_lines.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("- "):
            if in_ordered:
                body_lines.append("</ol>")
                in_ordered = False
            if not in_list:
                body_lines.append("<ul>")
                in_list = True
            body_lines.append(f"<li>{_inline_markdown_to_html(line[2:])}</li>")
        elif ". " in line and line.split(". ", 1)[0].isdigit():
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            if not in_ordered:
                body_lines.append("<ol>")
                in_ordered = True
            _, item = line.split(". ", 1)
            body_lines.append(f"<li>{_inline_markdown_to_html(item)}</li>")
        elif not line.strip():
            close_lists()
        else:
            close_lists()
            body_lines.append(f"<p>{_inline_markdown_to_html(line)}</p>")

    close_lists()
    return (
        "<div style=\"font-family: Arial, sans-serif; padding: 20px; line-height: 1.45;\">"
        "<div style=\"border-bottom: 2px solid #333; margin-bottom: 18px;\">"
        f"<p><b>{html.escape(report['report_header']['status'])}</b></p>"
        "</div>"
        + "\n".join(body_lines)
        + "<hr><p style=\"font-size: 0.9em; color: #555;\">Generated by a deterministic grounded-report template. "
        "This is not an autonomous radiology diagnosis.</p></div>"
    )


def _inline_markdown_to_html(text: str) -> str:
    escaped = html.escape(text)
    while "**" in escaped:
        escaped = escaped.replace("**", "<b>", 1)
        if "**" in escaped:
            escaped = escaped.replace("**", "</b>", 1)
        else:
            escaped += "</b>"
    return escaped.replace("`", "")


def build_llm_style_prompt(report: dict[str, Any]) -> str:
    """Return a prompt for optional style-only rewriting.

    This prompt is intentionally strict: an LLM may improve grammar, but it is
    not allowed to add clinical facts. Use the deterministic report when safety
    matters more than prose style.
    """

    return (
        "Rewrite the following AI draft MRI report for clarity only. Do not add, remove, or infer any "
        "clinical findings. Keep every 'not assessed' limitation. Return valid JSON with the same keys "
        "and values unless a sentence is grammatically rephrased without changing meaning.\n\n"
        f"{json.dumps(report, indent=2)}"
    )


def _load_prediction_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prediction = data.get("label") or data.get("prediction") or data.get("subtype")
    confidence = data.get("confidence") or data.get("specialist_confidence")
    model_used = data.get("specialist_checkpoint") or data.get("model_used")
    return {
        "prediction": prediction,
        "confidence": confidence,
        "model_used": model_used,
        "source_image": data.get("source_image"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a grounded MRI report from classifier output.")
    parser.add_argument("--prediction", help="Classifier label, e.g. glioma, MildDemented, tumor_glioma.")
    parser.add_argument("--confidence", help="Classifier confidence, e.g. 97.2%% or 0.972.")
    parser.add_argument("--model-used")
    parser.add_argument("--source-image")
    parser.add_argument("--prediction-json", type=Path, help="JSON output from src.hierarchical_inference.")
    parser.add_argument("--patient-name", default="Not provided")
    parser.add_argument("--patient-id", default="Not provided")
    parser.add_argument("--dob", default="Not provided")
    parser.add_argument("--reason", default="Not provided")
    parser.add_argument("--history", default="Not provided")
    parser.add_argument("--comparison", default="None")
    parser.add_argument("--technique", default="MRI brain protocol not specified")
    parser.add_argument("--contrast", default="Not specified")
    parser.add_argument("--format", choices=["json", "markdown", "html"], default="json")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_payload: dict[str, Any] = {}
    if args.prediction_json:
        prediction_payload = _load_prediction_json(args.prediction_json)

    prediction = args.prediction or prediction_payload.get("prediction")
    if not prediction:
        raise SystemExit("--prediction or --prediction-json is required")

    report = build_grounded_report(
        prediction=prediction,
        confidence=args.confidence or prediction_payload.get("confidence"),
        patient_info={"name": args.patient_name, "patient_id": args.patient_id, "dob": args.dob},
        exam_details={
            "reason": args.reason,
            "history": args.history,
            "comparison": args.comparison,
            "technique": args.technique,
            "contrast": args.contrast,
        },
        model_used=args.model_used or prediction_payload.get("model_used"),
        source_image=args.source_image or prediction_payload.get("source_image"),
    )

    if args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "markdown":
        output = render_report_markdown(report)
    else:
        output = render_report_html(report)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
