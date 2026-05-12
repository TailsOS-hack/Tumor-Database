# CNN Publication Audit

The CNN classifiers are the publication-grade models in this project, but their strict-test metrics are near-perfect. That is useful only if the manuscript clearly documents leakage controls, overfitting checks, and source-bias limitations.

## Current Local Status

Local audit report: `training_logs/publication_audit/local_summary/audit_report.md`

Full Kaggle audit summary: `training_logs/publication_audit/cnn_publication_audit_summary.json`

Status after the first full Kaggle audit: `blocking_leakage_risk`

Blocking finding:

- Exact cross-split image hash overlaps: 782 rows, representing 228 unique exact image hashes.
- Perceptual cross-split dHash overlaps: 22,537 rows, representing 3,372 unique perceptual hashes that need manual review.
- Missing manifest files: 0.

Known local limitations:

- The sparse local checkout does not contain full image data for image-hash checks.
- The sparse local checkout does not contain the full CNN training histories.
- The binary router is vulnerable to source/dataset bias because tumor and dementia images originate from different datasets.

## Remote Kaggle Audit and Retraining

Script: `notebooks/kaggle_cnn_publication_audit_kernel.py`

Kaggle kernel id: `armankazi/tumor-cnn-publication-audit`

The first full remote run confirmed exact duplicate leakage, so current and regularized checkpoints from that run should not replace the published model files.

The next remote run uses the updated manifest builder, which groups exact duplicate SHA-256 image hashes into one split before training. Its output paths use `dedup` names so the leaked-run evidence is preserved.

The remote run is designed to:

- Rebuild the strict manifest.
- Compute exact SHA-256 duplicate overlap across train/validation/test.
- Compute perceptual dHash overlap across train/validation/test.
- Evaluate existing checkpoints on train, validation, and test splits when LFS checkpoints are available.
- Run a regularized CNN suite with label smoothing, random erasing, stronger weight decay, and early stopping.
- Package audit JSON/Markdown, metrics, confusion matrices, histories, and retrained `.pt` checkpoints.

## Decision Rule

Do not replace the current `.pt` checkpoints just because a regularized run finishes. Import the Kaggle artifacts first, compare strict-test accuracy, train/validation gaps, and hash-overlap results, then replace checkpoints only if exact cross-split hash overlap is zero and the regularized models preserve strong test performance while reducing reviewer risk.
