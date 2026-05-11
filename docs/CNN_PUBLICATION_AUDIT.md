# CNN Publication Audit

The CNN classifiers are the publication-grade models in this project, but their strict-test metrics are near-perfect. That is useful only if the manuscript clearly documents leakage controls, overfitting checks, and source-bias limitations.

## Current Local Status

Local audit report: `training_logs/publication_audit/local_summary/audit_report.md`

Status: `reviewer_risk_needs_documentation`

Known local limitations:

- The sparse local checkout does not contain full image data for image-hash checks.
- The sparse local checkout does not contain the full CNN training histories.
- The binary router is vulnerable to source/dataset bias because tumor and dementia images originate from different datasets.

## Remote Kaggle Audit

Script: `notebooks/kaggle_cnn_publication_audit_kernel.py`

Kaggle kernel id: `armankazi/tumor-cnn-publication-audit`

The remote run is designed to:

- Rebuild the strict manifest.
- Compute exact SHA-256 duplicate overlap across train/validation/test.
- Compute perceptual dHash overlap across train/validation/test.
- Evaluate existing checkpoints on train, validation, and test splits when LFS checkpoints are available.
- Run a regularized CNN suite with label smoothing, random erasing, stronger weight decay, and early stopping.
- Package audit JSON/Markdown, metrics, confusion matrices, histories, and retrained `.pt` checkpoints.

## Decision Rule

Do not replace the current `.pt` checkpoints just because a regularized run finishes. Import the Kaggle artifacts first, compare strict-test accuracy, train/validation gaps, and hash-overlap results, then replace checkpoints only if the regularized models preserve strong test performance while reducing reviewer risk.
