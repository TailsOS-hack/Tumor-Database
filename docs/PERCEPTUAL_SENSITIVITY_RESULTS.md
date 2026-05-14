# Perceptual-Hash Sensitivity Results

This is the conservative robustness run from the corrected Kaggle version 2 job. The manifest groups exact SHA-256 duplicates and identical audit-compatible dHash fingerprints into the same split. It is stricter than the accepted de-duplicated baseline and is intended as sensitivity evidence, not as the default checkpoint set.

Audit summary:

- Manifest rows: 51,023.
- Split counts: train 34,759, validation 5,364, test 10,900.
- Exact SHA-256 cross-split overlaps: 0.
- Perceptual dHash cross-split overlaps: 0.
- Missing files: 0.
- Train/validation overfitting gap flags: 0.

| Model | Status | Accuracy | Macro F1 | Weighted F1 | Metrics Path |
| --- | --- | ---: | ---: | ---: | --- |
| binary | complete | 1.0000 | 1.0000 | 1.0000 | `training_logs/experiments_perceptual_regularized/binary/20260513_155157/test/metrics.json` |
| tumor | complete | 0.9539 | 0.9535 | 0.9530 | `training_logs/experiments_perceptual_regularized/tumor/20260513_162514/test/metrics.json` |
| dementia | complete | 0.9975 | 0.9976 | 0.9975 | `training_logs/experiments_perceptual_regularized/dementia/20260513_163515/test/metrics.json` |
| hierarchical | complete | 0.9894 | 0.9755 | 0.9893 | `training_logs/experiments_perceptual_regularized/hierarchical/test_evaluation/metrics.json` |
| eight_class | complete | 0.9906 | 0.9740 | 0.9904 | `training_logs/experiments_perceptual_regularized/eight_class/20260513_170749/test/metrics.json` |

Decision: keep the accepted exact-deduplicated checkpoints as the primary model files. Use this result as a conservative sensitivity analysis showing that performance remains high even when exact dHash groups are prevented from crossing splits.
