# De-duplicated CNN Publication Summary

These are the accepted strict-test CNN results from the Kaggle de-duplicated, regularized retrain. The dataset images are brain MRI scans. Exact SHA-256 duplicate leakage was removed before this run by assigning duplicate image groups to a single split.

| Model | Status | Accuracy | Macro F1 | Weighted F1 | Metrics Path |
| --- | --- | ---: | ---: | ---: | --- |
| binary | complete | 1.0000 | 1.0000 | 1.0000 | `training_logs/experiments_dedup_regularized/binary/20260512_013727/test/metrics.json` |
| tumor | complete | 0.9792 | 0.9791 | 0.9793 | `training_logs/experiments_dedup_regularized/tumor/20260512_021514/test/metrics.json` |
| dementia | complete | 0.9991 | 0.9991 | 0.9991 | `training_logs/experiments_dedup_regularized/dementia/20260512_022515/test/metrics.json` |
| hierarchical | complete | 0.9963 | 0.9891 | 0.9963 | `training_logs/experiments_dedup_regularized/hierarchical/test_evaluation/metrics.json` |
| eight_class | complete | 0.9972 | 0.9905 | 0.9972 | `training_logs/experiments_dedup_regularized/eight_class/20260512_025559/test/metrics.json` |

Audit status:

- Exact cross-split SHA-256 overlaps: 0.
- Missing manifest files: 0.
- Perceptual dHash overlaps remain and should be discussed as a limitation.
- Train/validation overfitting gap flags: 0.
