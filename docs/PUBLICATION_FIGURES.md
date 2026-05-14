# Publication Figure Manifest

This manifest maps the manuscript draft to the exact figure assets already produced by the audited runs.

## Core Figures

| Figure | Purpose | Source artifact |
| --- | --- | --- |
| Figure 1 | Dataset and split workflow, including exact SHA-256 grouping and dHash sensitivity audit | To be drawn from `docs/CNN_PUBLICATION_AUDIT.md`, `docs/DEDUP_RETRAIN_SUMMARY.md`, and `docs/PERCEPTUAL_SENSITIVITY_SUMMARY.md` |
| Figure 2 | Architecture comparison: hierarchical router plus specialists versus single 8-class CNN | To be drawn from `docs/ML_EXECUTION_FLOW.md` and `docs/PUBLICATION_RESULTS_TABLES.md` |
| Figure 3A | Accepted exact-deduplicated hierarchical CNN confusion matrix | `training_logs/experiments_dedup_regularized/hierarchical/test_evaluation/confusion_matrix.png` |
| Figure 3B | Accepted exact-deduplicated single 8-class CNN confusion matrix | `training_logs/experiments_dedup_regularized/eight_class/20260512_025559/test/confusion_matrix.png` |
| Figure 4A | Conservative dHash sensitivity hierarchical CNN confusion matrix | `training_logs/experiments_perceptual_regularized/hierarchical/test_evaluation/confusion_matrix.png` |
| Figure 4B | Conservative dHash sensitivity single 8-class CNN confusion matrix | `training_logs/experiments_perceptual_regularized/eight_class/20260513_170749/test/confusion_matrix.png` |
| Figure 5 | CNN versus multimodal VLM comparison | `docs/publication_cnn_results.csv` and `docs/publication_vlm_results.csv` |
| Figure 6 | Calibration and confidence reliability for the accepted CNN checkpoints | `training_logs/publication_evidence/*/calibration.png` and `training_logs/publication_evidence/*/confidence_histogram.png` |
| Figure 7 | One-vs-rest ROC and precision-recall evidence for the accepted CNN checkpoints | `training_logs/publication_evidence/*/roc_curves.png` and `training_logs/publication_evidence/*/precision_recall_curves.png` |

## Supporting Confusion Matrices

| Model | Accepted exact-deduplicated | Conservative dHash sensitivity |
| --- | --- | --- |
| Binary router | `training_logs/experiments_dedup_regularized/binary/20260512_013727/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/binary/20260513_155157/test/confusion_matrix.png` |
| Tumor specialist | `training_logs/experiments_dedup_regularized/tumor/20260512_021514/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/tumor/20260513_162514/test/confusion_matrix.png` |
| Dementia specialist | `training_logs/experiments_dedup_regularized/dementia/20260512_022515/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/dementia/20260513_163515/test/confusion_matrix.png` |

## Export Notes

- Existing confusion matrix images are 1440 x 1080 PNG files, suitable for draft review.
- Publication evidence plots are generated from the accepted checkpoint probability outputs in `training_logs/publication_evidence/`.
- For journal submission, export final panels at the journal-required resolution and combine Figure 3 and Figure 4 into labeled multi-panel images.
- Keep `docs/PUBLICATION_RESULTS_TABLES.md` and `docs/PUBLICATION_EVIDENCE_RESULTS.md` as the source of truth for numeric values in figure captions.
