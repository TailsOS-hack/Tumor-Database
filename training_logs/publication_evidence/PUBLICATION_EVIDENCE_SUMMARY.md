# Publication Evidence Summary

This bundle was generated from existing checkpoints. It does not retrain models.

| Model | N | Accuracy | Routed Acc. | Macro F1 | ECE | Brier | ROC AUC | AP | Mean Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | 10251 | 1.0000 | NA | 1.0000 | 0.1189 | 0.0325 | 1.0000 | 1.0000 | 0.8811 |
| tumor | 1445 | 0.9792 | NA | 0.9791 | 0.0372 | 0.0409 | 0.9975 | 0.9875 | 0.9420 |
| dementia | 8806 | 0.9991 | NA | 0.9991 | 0.0386 | 0.0041 | 1.0000 | 1.0000 | 0.9605 |
| eight_class | 10251 | 0.9972 | NA | 0.9905 | 0.1327 | 0.0285 | 0.9979 | 0.9969 | 0.8647 |
| hierarchical | 10251 | 0.9963 | 0.9963 | 0.9891 | 0.1526 | 0.0474 | 0.9968 | 0.9907 | 0.8437 |

## Artifacts

Each model folder contains:

- `probabilities.csv`
- `calibration_bins.csv`
- `metrics.json`
- `calibration.png`
- `confidence_histogram.png`
- `roc_curves.png`
- `precision_recall_curves.png`
