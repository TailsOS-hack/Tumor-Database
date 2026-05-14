# Publication Notes Template

Use this file as the running methods/results record. Fill it only from strict test outputs produced by `src.experiment_pipeline`.

## Study Goal

Compare a hierarchical MRI classifier against a single 8-class classifier for tumor and dementia image categorization, then evaluate whether multimodal LLMs improve report metadata extraction or report generation. The image datasets used here are brain MRI scans, not CT scans.

## Dataset and Splits

| Dataset | Classes | Split Source | Notes |
| --- | --- | --- | --- |
| Brain tumor MRI | glioma, meningioma, notumor, pituitary | Official Training/Testing folders plus exact-duplicate grouped split units | Kaggle Brain Tumor MRI Dataset by Masoud Nickparvar; official Testing folder held out when possible; duplicate groups never cross splits |
| Dementia MRI | MildDemented, ModerateDemented, NonDemented, VeryMildDemented | Deterministic stratified split plus exact-duplicate grouped split units | Kaggle Alzheimer's Disease Multiclass Images Dataset by Aryan Singhal; augmented/up-sampled derivative source chain; no official dementia holdout was provided |

Manifest path: `training_logs/splits/strict_manifest.csv`

Dataset provenance and citation notes: `docs/DATASET_PROVENANCE.md`

## Architectures

| Experiment | Router | Specialist / Head | Output Space | Checkpoint |
| --- | --- | --- | --- | --- |
| Hierarchical | ResNet50 binary router | EfficientNet-B3 tumor, MobileNetV3 dementia | Binary route plus 4-class specialists | `models/binary_router.pt`, specialists |
| Single model | None | EfficientNet-B3 | 8 classes | `models/single_8class_classifier.pt` |
| Multimodal LLM | Qwen/Qwen2.5-VL-3B-Instruct | LoRA adapter | Structured JSON labels/report metadata | `models/multimodal/qwen25vl_3b_mri_lora/` |

## Accepted De-duplicated Strict-Test Results

| Model | Accuracy | Macro F1 | Weighted F1 | Key Failure Modes | Metrics Path |
| --- | ---: | ---: | ---: | --- | --- |
| Binary router | 1.0000 | 1.0000 | 1.0000 | Possible tumor-vs-dementia source bias should be discussed | `training_logs/experiments_dedup_regularized/binary/20260512_013727/test/metrics.json` |
| Tumor specialist | 0.9792 | 0.9791 | 0.9793 | Subtype errors increased after exact duplicate leakage removal | `training_logs/experiments_dedup_regularized/tumor/20260512_021514/test/metrics.json` |
| Dementia specialist | 0.9991 | 0.9991 | 0.9991 | Dementia split may be optimistic because no official holdout was provided | `training_logs/experiments_dedup_regularized/dementia/20260512_022515/test/metrics.json` |
| Hierarchical end-to-end | 0.9963 | 0.9891 | 0.9963 | Inherits source-bias and specialist risks | `training_logs/experiments_dedup_regularized/hierarchical/test_evaluation/metrics.json` |
| Single 8-class | 0.9972 | 0.9905 | 0.9972 | Slightly higher accuracy than hierarchical; still needs external validation | `training_logs/experiments_dedup_regularized/eight_class/20260512_025559/test/metrics.json` |

Former near-perfect CNN results under `training_logs/experiments/` are retained for provenance but should not be used as final publication claims because the first full Kaggle audit found exact cross-split duplicate leakage.

## Conservative Perceptual-Hash Sensitivity Results

This run uses `training_logs/splits/perceptual_strict_manifest.csv`, which groups exact SHA-256 duplicates and identical audit-compatible dHash fingerprints into the same split. It is a robustness analysis for the near-duplicate concern, not the default model checkpoint set.

| Model | Accuracy | Macro F1 | Weighted F1 | Interpretation | Metrics Path |
| --- | ---: | ---: | ---: | --- | --- |
| Binary router | 1.0000 | 1.0000 | 1.0000 | Still likely source-separable | `training_logs/experiments_perceptual_regularized/binary/20260513_155157/test/metrics.json` |
| Tumor specialist | 0.9539 | 0.9535 | 0.9530 | Conservative tumor estimate under dHash grouping | `training_logs/experiments_perceptual_regularized/tumor/20260513_162514/test/metrics.json` |
| Dementia specialist | 0.9975 | 0.9976 | 0.9975 | Robust under dHash grouping | `training_logs/experiments_perceptual_regularized/dementia/20260513_163515/test/metrics.json` |
| Hierarchical end-to-end | 0.9894 | 0.9755 | 0.9893 | Conservative support for the accepted baseline | `training_logs/experiments_perceptual_regularized/hierarchical/test_evaluation/metrics.json` |
| Single 8-class | 0.9906 | 0.9740 | 0.9904 | Conservative support for the accepted baseline | `training_logs/experiments_perceptual_regularized/eight_class/20260513_170749/test/metrics.json` |

## Multimodal LLM Benchmark

| Candidate | Size | Runtime | Quantization | Strict JSON Rate | Label Accuracy | Report Quality Notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1719 | Best zero-shot VLM, still weak |
| Qwen/Qwen2-VL-2B-Instruct | 2B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1250 | Smaller Qwen baseline |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1750 | Best zero-shot in batch 2, still weak |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | Kaggle 2x T4 | 4-bit | 0.7292 | 0.0833 | Dependency fixed in batch 2; JSON and accuracy weak |
| Qwen/Qwen2.5-VL-3B-Instruct + LoRA | 3B | Kaggle 2x T4 | 4-bit LoRA | 1.0000 | 0.3125 | Improved after 256-example LoRA but still collapsed labels |
| Qwen/Qwen2.5-VL-7B-Instruct hierarchical | 7B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.2250 | Broad tumor-vs-dementia routing improved to 0.8750, but subtype collapse remained |
| llava-hf/llava-v1.6-34b-hf | 34B | Kaggle 2x T4 | 4-bit | N/A | N/A | Skipped because memory was insufficient |

## CNN Publication Audit

The CNN metrics are strong enough that the paper must proactively address leakage, source bias, and overfitting. The local audit summary currently reports `reviewer_risk_needs_documentation` rather than a blocking failure because the sparse checkout does not contain the full image set or training histories.

Local audit artifact: `training_logs/publication_audit/local_summary/audit_report.md`

Full initial Kaggle audit artifact: `training_logs/publication_audit/cnn_publication_audit_summary.json`

Accepted de-duplicated audit artifact: `training_logs/publication_audit/cnn_dedup_retrain_summary.json`

Perceptual sensitivity audit artifact: `training_logs/publication_audit/cnn_perceptual_sensitivity_summary.json`

Remote audit/retraining script: `notebooks/kaggle_cnn_publication_audit_kernel.py`

The first full Kaggle audit found a blocking leakage risk:

- Exact cross-split SHA-256 overlaps: 782 rows, 228 unique hashes.
- Perceptual dHash cross-split overlaps: 22,537 rows, 3,372 unique hashes.
- Regularized retraining showed no train/validation gap flags, but those checkpoints were still trained on the leaky manifest and should not replace the current model files.

The accepted de-duplicated Kaggle rerun found:

- Exact cross-split SHA-256 overlaps: 0.
- Missing manifest files: 0.
- Perceptual dHash cross-split overlaps: 22,304 rows, 3,290 hashes.
- Train/validation gap flags: 0.
- Regularized retraining used label smoothing, random erasing, stronger weight decay, and early stopping.
- The accepted `.pt` checkpoint files now come from the de-duplicated regularized run.

The corrected perceptual-hash sensitivity rerun found:

- Exact cross-split SHA-256 overlaps: 0.
- Perceptual dHash cross-split overlaps: 0.
- Missing manifest files: 0.
- Train/validation gap flags: 0.
- Hierarchical strict-test accuracy: 0.9894.
- Single 8-class strict-test accuracy: 0.9906.

## Methods Notes

- Report all final metrics from the strict test split.
- Include confusion matrices for every classifier and end-to-end hierarchical inference.
- Keep validation metrics separate from test metrics.
- Record random seed, epochs, image size, optimizer, learning rate, batch size, GPU type, and checkpoint hash for each run.
- Treat any accuracy above 0.995 as a reviewer-risk flag that needs leakage/source-bias discussion, not as self-validating evidence.
- State that the accepted CNN run fixed exact duplicate leakage, and the corrected perceptual-hash sensitivity run removed dHash overlap as a robustness check.
- For LoRA, report base model, adapter rank, target modules, quantization, training examples, validation examples, and adapter path.
- Current multimodal conclusion: VLMs should not replace the CNN classifiers for image labeling; treat them as report/metadata assistants unless a redesigned task produces materially stronger strict-test accuracy.
- Dataset citation conclusion: cite the Kaggle source datasets and the dementia upstream source chain; re-check current Kaggle license fields immediately before submission.

## Rationale Draft

The hierarchical approach is expected to reduce subtype confusion by first separating broad image domains, then applying specialized classifiers trained on narrower label spaces. The 8-class baseline tests whether a single shared visual representation can learn both datasets without routing errors. The multimodal LLM experiments are treated separately because they are evaluated for structured metadata/report generation and should not replace strict visual classifier metrics without evidence.

## Publication Package Gate

Run the lightweight submission gate before sharing the manuscript package:

```bash
python3 scripts/check_publication_package.py
```

The gate writes `docs/PUBLICATION_SUBMISSION_STATUS.md`, verifies required tables, figures, audit artifacts, probability evidence, source scripts, and grounded-report documentation, and keeps active warnings visible. It does not retrain models.
