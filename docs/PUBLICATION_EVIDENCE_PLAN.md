# Publication Evidence Job

This job evaluates the already accepted checkpoints and creates manuscript support artifacts. It does not retrain models.

## Purpose

The current CNN metrics are strong enough for the primary result table, but a publication draft still benefits from:

- probability CSVs for reproducibility
- calibration curves and expected calibration error
- confidence histograms for correct vs incorrect predictions
- one-vs-rest ROC curves
- precision-recall curves
- a compact evidence summary for figure captions

## Command

```bash
python scripts/build_publication_evidence.py \
  --output-dir training_logs/publication_evidence \
  --batch-size 64 \
  --num-workers 2
```

The Kaggle version runs the same command and zips the resulting `tumor_publication_evidence_outputs` folder.

## Expected Outputs

For each of `binary`, `tumor`, `dementia`, `eight_class`, and `hierarchical`:

- `probabilities.csv`
- `calibration_bins.csv`
- `metrics.json`
- `calibration.png`
- `confidence_histogram.png`
- `roc_curves.png`
- `precision_recall_curves.png`

Bundle-level outputs:

- `publication_evidence_summary.json`
- `PUBLICATION_EVIDENCE_SUMMARY.md`
- `tumor_publication_evidence_outputs.zip` on Kaggle

## Publication Use

Use these outputs to build final manuscript figures after the evidence zip is downloaded and imported into the repo.
