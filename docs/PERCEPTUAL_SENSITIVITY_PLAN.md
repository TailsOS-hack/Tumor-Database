# Perceptual-Hash Sensitivity Plan

The accepted CNN baseline has zero exact SHA-256 cross-split overlap, but the audit still reports identical dHash fingerprints across splits. This is not automatically leakage because dHash is a coarse visual fingerprint and can collide across distinct MRI slices, especially in the dementia classes. It is still a publication risk that needs a sensitivity check.

## Local Overlap Summary

From `training_logs/publication_audit/dedup_regularized/perceptual_hash_cross_split_overlaps.csv`:

| Item | Count |
| --- | ---: |
| Overlap rows | 22,304 |
| Unique dHash groups | 3,290 |
| Groups crossing tumor/dementia domains | 1 |
| Groups crossing eight-class labels | 1,421 |
| Rows in cross-label groups | 15,256 |
| Groups with both test and train rows | 2,457 |
| Groups with both test and validation rows | 849 |
| Largest dHash group | 269 rows |

Most overlap rows are dementia images. The large cross-label groups are evidence that exact dHash grouping is stricter than true duplicate removal and may group visually similar but clinically different MRI slices. That is why the next run is a sensitivity experiment, not an automatic replacement for the accepted baseline.

## Remote Sensitivity Run

Kaggle script: `notebooks/kaggle_cnn_perceptual_sensitivity_kernel.py`

Kernel id: `armankazi/tumor-cnn-perceptual-sensitivity`

The script will:

- Create `training_logs/splits/perceptual_strict_manifest.csv` with exact SHA-256 and identical audit-compatible dHash groups assigned to a single split.
- Audit the perceptual manifest before training.
- Retrain the regularized CNN suite on the perceptual manifest.
- Re-run publication audit on the perceptual run.
- Package metrics, confusion matrices, histories, manifest, audit files, and sensitivity checkpoint files.

## Decision Rule

If the perceptual sensitivity run stays near the accepted de-duplicated baseline, use it as a robustness paragraph and keep the current exact-deduplicated checkpoints as the main model files. If performance drops sharply, report the perceptual run as the conservative estimate and consider replacing the main publication claims with the stricter sensitivity results.
