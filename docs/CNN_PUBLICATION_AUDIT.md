# CNN Publication Audit

The CNN classifiers are the publication-grade models in this project, but their strict-test metrics are near-perfect. That is useful only if the manuscript clearly documents leakage controls, overfitting checks, and source-bias limitations.

## Current Publication Status

The accepted CNN baseline is the de-duplicated, regularized Kaggle retrain. These are brain MRI images, not CT images.

Accepted summary: `training_logs/publication_audit/cnn_dedup_retrain_summary.json`

Accepted strict-test results: `docs/DEDUP_PUBLICATION_RESULTS.md`

Accepted checkpoint files:

- `models/binary_router.pt`
- `models/brain_tumor_classifier.pt`
- `models/alzheimers_classifier.pt`
- `models/single_8class_classifier.pt`

The first full Kaggle audit remains preserved because it found a real blocker:

- Exact cross-split image hash overlaps: 782 rows, representing 228 unique exact image hashes.
- Perceptual cross-split dHash overlaps: 22,537 rows, representing 3,372 unique perceptual hashes that need manual review.
- Missing manifest files: 0.

The de-duplicated retrain fixed the blocking issue:

- Exact cross-split SHA-256 overlaps: 0.
- Missing manifest files: 0.
- Perceptual dHash overlaps: 22,304 rows, representing 3,290 perceptual hashes that still need reviewer-facing discussion and optional sensitivity analysis.
- Train/validation overfitting gap flags: 0.
- Risk status: `reviewer_risk_needs_documentation`, not `blocking_leakage_risk`.

## Remote Kaggle Audit and Retraining

Script: `notebooks/kaggle_cnn_publication_audit_kernel.py`

Kaggle kernel id: `armankazi/tumor-cnn-publication-audit`

The first full remote run confirmed exact duplicate leakage, so checkpoints from that run did not replace the published model files.

The accepted remote run used the updated manifest builder, which groups exact duplicate SHA-256 image hashes into one split before training. Its output paths use `dedup` names so the leaked-run evidence is preserved.

The remote run is designed to:

- Rebuild the strict manifest.
- Compute exact SHA-256 duplicate overlap across train/validation/test.
- Compute perceptual dHash overlap across train/validation/test.
- Evaluate existing checkpoints on train, validation, and test splits when LFS checkpoints are available.
- Run a regularized CNN suite with label smoothing, random erasing, stronger weight decay, and early stopping.
- Package audit JSON/Markdown, metrics, confusion matrices, histories, and retrained `.pt` checkpoints.

## Accepted Metrics

| Model | Accuracy | Macro F1 | Weighted F1 | Notes |
| --- | ---: | ---: | ---: | --- |
| Binary router | 1.0000 | 1.0000 | 1.0000 | Discuss source/dataset bias explicitly |
| Tumor specialist | 0.9792 | 0.9791 | 0.9793 | More realistic after leakage removal |
| Dementia specialist | 0.9991 | 0.9991 | 0.9991 | Needs external-validity caveat |
| Hierarchical end-to-end | 0.9963 | 0.9891 | 0.9963 | Accepted main architecture |
| Single 8-class baseline | 0.9972 | 0.9905 | 0.9972 | Slightly higher accuracy than hierarchical |

## Decision

The de-duplicated regularized checkpoints are accepted as the repo's current model files because exact cross-split hash overlap is zero, strict-test performance remains strong, and no train/validation overfitting gap flags were found. The paper should still include a limitation paragraph on source bias, near-duplicate perceptual-hash overlap, and the need for external validation.
