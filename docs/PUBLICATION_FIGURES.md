# Publication Figure Manifest

This manifest maps the manuscript draft to the final generated figure files and the exact source artifacts used to build them. Regenerate the figure package with `python3 scripts/build_publication_figures.py`.

## Core Figures

| Figure | Purpose | Final artifact | Source evidence |
| --- | --- | --- | --- |
| Figure 1 | Dataset and split workflow, including exact SHA-256 grouping and dHash sensitivity audit | `docs/figures/figure1_workflow.png` | `docs/CNN_PUBLICATION_AUDIT.md`, `docs/DEDUP_RETRAIN_SUMMARY.md`, `docs/PERCEPTUAL_SENSITIVITY_SUMMARY.md`, `docs/publication_audit_checks.csv` |
| Figure 2 | Architecture comparison: hierarchical router plus specialists versus single 8-class CNN | `docs/figures/figure2_architecture.png` | `docs/ML_EXECUTION_FLOW.md`, `docs/PUBLICATION_RESULTS_TABLES.md` |
| Figure 3 | Accepted exact-deduplicated confusion matrices for tumor, dementia, hierarchical, and single 8-class CNNs | `docs/figures/figure3_exact_dedup_confusion.png` | `training_logs/experiments_dedup_regularized/*/confusion_matrix.png` |
| Figure 4 | Conservative dHash sensitivity confusion matrices for tumor, dementia, hierarchical, and single 8-class CNNs | `docs/figures/figure4_dhash_sensitivity_confusion.png` | `training_logs/experiments_perceptual_regularized/*/confusion_matrix.png` |
| Figure 5 | CNN versus multimodal VLM comparison | `docs/figures/figure5_cnn_vlm_comparison.png` | `docs/publication_cnn_results.csv` and `docs/publication_vlm_results.csv` |
| Figure 6 | Calibration and confidence reliability for the accepted CNN checkpoints | `docs/figures/figure6_calibration_confidence.png` | `training_logs/publication_evidence/*/calibration.png` and `training_logs/publication_evidence/*/confidence_histogram.png` |
| Figure 7 | One-vs-rest ROC and precision-recall evidence for the accepted CNN checkpoints | `docs/figures/figure7_roc_pr_curves.png` | `training_logs/publication_evidence/*/roc_curves.png` and `training_logs/publication_evidence/*/precision_recall_curves.png` |

## Supporting Confusion Matrices

| Model | Accepted exact-deduplicated | Conservative dHash sensitivity |
| --- | --- | --- |
| Binary router | `training_logs/experiments_dedup_regularized/binary/20260512_013727/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/binary/20260513_155157/test/confusion_matrix.png` |
| Tumor specialist | `training_logs/experiments_dedup_regularized/tumor/20260512_021514/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/tumor/20260513_162514/test/confusion_matrix.png` |
| Dementia specialist | `training_logs/experiments_dedup_regularized/dementia/20260512_022515/test/confusion_matrix.png` | `training_logs/experiments_perceptual_regularized/dementia/20260513_163515/test/confusion_matrix.png` |

## Export Notes

- Existing confusion matrix images are 1440 x 1080 PNG files, suitable for draft review.
- Final combined figure PNG files are available in `docs/figures/`.
- Draft captions are available in `docs/FIGURE_CAPTIONS.md`.
- Publication evidence plots are generated from the accepted checkpoint probability outputs in `training_logs/publication_evidence/`.
- For journal submission, export final panels at the journal-required resolution and adjust lettering or panel sizing to match the target journal style guide.
- Keep `docs/PUBLICATION_RESULTS_TABLES.md` and `docs/PUBLICATION_EVIDENCE_RESULTS.md` as the source of truth for numeric values in figure captions.
