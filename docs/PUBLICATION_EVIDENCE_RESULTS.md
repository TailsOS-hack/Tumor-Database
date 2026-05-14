# Publication Evidence Results

The publication evidence bundle was generated on Kaggle version 3 from commit `790bd37`. It evaluates existing checkpoints only; no retraining was performed.

Primary artifact directory:

`training_logs/publication_evidence/`

## Summary Metrics

| Model | N | Accuracy | Routed Acc. | Macro F1 | ECE | Brier | ROC AUC | AP | Mean Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Binary router | 10,251 | 1.0000 | NA | 1.0000 | 0.1189 | 0.0325 | 1.0000 | 1.0000 | 0.8811 |
| Tumor specialist | 1,445 | 0.9792 | NA | 0.9791 | 0.0372 | 0.0409 | 0.9975 | 0.9875 | 0.9420 |
| Dementia specialist | 8,806 | 0.9991 | NA | 0.9991 | 0.0386 | 0.0041 | 1.0000 | 1.0000 | 0.9605 |
| Single 8-class CNN | 10,251 | 0.9972 | NA | 0.9905 | 0.1327 | 0.0285 | 0.9979 | 0.9969 | 0.8647 |
| Hierarchical CNN | 10,251 | 0.9963 | 0.9963 | 0.9891 | 0.1526 | 0.0474 | 0.9968 | 0.9907 | 0.8437 |

## Artifact Map

Each model folder contains:

- `probabilities.csv`
- `calibration_bins.csv`
- `metrics.json`
- `calibration.png`
- `confidence_histogram.png`
- `roc_curves.png`
- `precision_recall_curves.png`

Model folders:

- `training_logs/publication_evidence/binary/`
- `training_logs/publication_evidence/tumor/`
- `training_logs/publication_evidence/dementia/`
- `training_logs/publication_evidence/eight_class/`
- `training_logs/publication_evidence/hierarchical/`

## Interpretation

The tumor and dementia specialists show strong discrimination and low calibration error. The single 8-class and hierarchical models keep high ROC AUC and AP, but their ECE values are higher than the domain specialists, so the paper should report confidence calibration as a limitation and avoid treating softmax confidence as calibrated clinical probability.
