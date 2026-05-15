# Full Manuscript Draft

Working title: Leakage-Audited Hierarchical and Single-Head CNNs for Brain MRI Tumor and Dementia Image Classification

Draft status: venue-neutral full manuscript draft. Adapt section headings, word count, reference style, and figure/table limits after selecting the target journal or conference.

## Author And Affiliation Placeholders

Authors: Arman Kazi, Mina [last name], and collaborators to be confirmed.

Affiliations: [Add institution, department, city, country].

Corresponding author: [Add name, email].

## Structured Abstract

Background: Public brain MRI image datasets are frequently used for tumor and dementia classification, but high reported performance can be inflated by duplicate leakage, augmented-image similarity, and source-specific artifacts. This study evaluates whether leakage-audited convolutional neural networks can provide reliable internal-test performance for combined tumor and dementia image classification, and whether multimodal vision-language models are competitive as direct image classifiers.

Methods: Brain tumor MRI images were sourced from the public Kaggle Brain Tumor MRI Dataset by Masoud Nickparvar. Dementia MRI images were sourced from the public Kaggle Alzheimer's Disease Multiclass Images Dataset by Aryan Singhal, an augmented derivative dataset with an upstream Kaggle source chain. Images were organized into four tumor classes and four dementia-stage classes. A hierarchical CNN pipeline, consisting of a ResNet50 binary tumor-versus-dementia router followed by EfficientNet-B3 tumor and MobileNetV3 dementia specialists, was compared with a single EfficientNet-B3 classifier trained across all eight classes. Splits were created before augmentation. Exact duplicate SHA-256 image groups were assigned to a single split for the accepted primary run. A stricter dHash-grouped sensitivity run was also performed. Metrics included strict-test accuracy, macro F1, weighted F1, confusion matrices, calibration, Brier score, ROC AUC, and average precision. Multimodal VLM baselines and LoRA adaptation were evaluated separately.

Results: The first full audit identified 782 exact cross-split SHA-256 overlap rows and was treated as a blocking leakage risk. After exact duplicate grouping and regularized retraining, exact cross-split overlap was reduced to 0. The accepted exact-deduplicated run achieved 0.9963 hierarchical CNN accuracy and 0.9972 single 8-class CNN accuracy on the strict test split. The tumor and dementia specialists reached 0.9792 and 0.9991 accuracy, respectively. In the conservative dHash sensitivity run, exact and dHash overlaps were both 0, with 0.9894 hierarchical accuracy and 0.9906 single 8-class accuracy. Probability-level evaluation showed high discrimination but imperfect calibration for the integrated models, with expected calibration error of 0.1327 for the single 8-class CNN and 0.1526 for the hierarchical CNN. Multimodal VLMs were not competitive as direct classifiers: the best LoRA-adapted direct VLM reached 0.3125 accuracy and the best hierarchical VLM diagnostic reached 0.2250 routed 8-class accuracy.

Conclusions: Leakage-audited CNNs achieved strong internal strict-test performance on the combined brain MRI dataset, and the single 8-class CNN slightly outperformed the hierarchical pipeline. The VLM experiments support using language models only as report or metadata assistants, not as direct MRI classifiers in this setting. The study should be presented as an internal public-dataset benchmark with explicit limitations: no patient-level metadata, no external validation cohort, source-domain bias risk, and imperfect calibration.

Keywords: brain MRI; brain tumor classification; dementia classification; convolutional neural networks; data leakage; medical imaging; calibration; vision-language models

## Highlights

- Exact duplicate leakage was detected in the initial strict split and corrected before final claims.
- The accepted exact-deduplicated single 8-class CNN reached 0.9972 strict-test accuracy.
- A conservative dHash sensitivity split retained strong performance at 0.9906 single 8-class accuracy.
- Multimodal VLMs and LoRA adaptation were much weaker than CNNs for direct MRI classification.
- Softmax scores should be reported as model confidence, not calibrated clinical probability.

## 1. Introduction

Deep learning systems for brain MRI classification are often evaluated on public image datasets that are convenient for rapid experimentation but difficult to interpret clinically. Public datasets may lack patient identifiers, acquisition metadata, scanner information, or external validation cohorts. These limitations are especially important when reported accuracy is very high, because near-perfect performance can reflect duplicate images, augmented-image similarity, or dataset-source artifacts rather than robust clinical generalization.

This project evaluates a combined brain MRI classification task spanning tumor and dementia image categories. The tumor branch contains glioma, meningioma, no-tumor, and pituitary classes. The dementia branch contains MildDemented, ModerateDemented, NonDemented, and VeryMildDemented classes. The study compares two CNN strategies. The first is a hierarchical design that routes each image through a binary tumor-versus-dementia classifier and then applies a domain-specific specialist. The second is a single 8-class classifier trained directly over all tumor and dementia labels.

The study was motivated by a practical question: whether a hierarchical architecture improves interpretability or accuracy relative to a single shared classifier when combining two public MRI datasets. A secondary question was whether multimodal vision-language models could replace or augment the CNN classifiers for direct image labeling. Because preliminary results were near-perfect, the project prioritized leakage audits, de-duplicated retraining, calibration evidence, and sensitivity analysis before defining publishable claims.

The main contribution is not a clinical deployment system. Instead, this manuscript presents a reproducible public-dataset benchmark with explicit leakage correction, robustness testing, and claim boundaries. The final paper should emphasize that external MRI validation and patient-level metadata are required before any clinical conclusion.

## 2. Materials And Methods

### 2.1 Datasets

Brain tumor MRI images were sourced from the public Kaggle Brain Tumor MRI Dataset by Masoud Nickparvar. The local dataset layout contains official `Training/` and `Testing/` folders for glioma, meningioma, no-tumor, and pituitary classes. The strict manifest includes 5,712 tumor training-pool rows and 1,311 tumor official-test rows.

Dementia MRI images were sourced from the public Kaggle Alzheimer's Disease Multiclass Images Dataset by Aryan Singhal. The local class counts match the data card: 10,000 MildDemented, 10,000 ModerateDemented, 12,800 NonDemented, and 11,200 VeryMildDemented images, 44,000 total. This dataset is an augmented and upsampled derivative dataset. Its data card cites UraninJo's Augmented Alzheimer MRI Dataset V2 as an upstream source, which in turn references Tourist55's Alzheimer's Dataset (4 class of Images). Dataset provenance and citation notes are maintained in `docs/DATASET_PROVENANCE.md`.

All images were treated as brain MRI images. The raw Kaggle images are not redistributed in this repository.

### 2.2 Split Construction And Leakage Controls

Splits were generated before augmentation. Tumor images from the official Testing folders were preferentially held out for the test split. Dementia images were deterministically stratified because the local dementia dataset does not contain an official test folder.

The initial strict split produced very high performance, so a publication audit was performed. That audit found 782 exact cross-split SHA-256 overlap rows, corresponding to 228 unique hashes. Because identical image pixels appeared across splits, those preliminary results were treated as non-publishable.

The accepted primary training manifest groups exact duplicate SHA-256 image hashes into a single split. This reduced exact cross-split overlap to 0 and missing manifest files to 0. Because perceptual dHash overlap remained in the exact-deduplicated audit, a conservative sensitivity manifest was also created by grouping exact hashes and identical audit-compatible dHash fingerprints into the same split. The sensitivity run produced 0 exact overlaps and 0 dHash overlaps.

Figure 1 summarizes the dataset splitting and leakage-audit workflow.

### 2.3 Image Preprocessing And Augmentation

Training augmentation was applied only to training images and included horizontal flips, rotations, contrast jitter, and random erasing in the regularized retraining runs. Validation and test transforms were limited to deterministic preprocessing, including resizing, tensor conversion, and normalization. Class-weighted cross-entropy and label smoothing were used during regularized retraining.

### 2.4 CNN Architectures

The hierarchical pipeline used three CNN checkpoints:

- Binary router: ResNet50 trained to classify tumor versus dementia.
- Tumor specialist: EfficientNet-B3 trained over glioma, meningioma, no-tumor, and pituitary.
- Dementia specialist: MobileNetV3-Large trained over MildDemented, ModerateDemented, NonDemented, and VeryMildDemented.

The comparison baseline used a single EfficientNet-B3 classifier trained over all eight tumor and dementia classes. Figure 2 summarizes the architecture comparison.

### 2.5 Training And Evaluation

Training used ImageNet-pretrained backbones, AdamW optimization, stronger weight decay, class weighting, label smoothing, random erasing, and early stopping. Heavy training and audit runs were performed on Kaggle GPU runners. Local commands are retained only for lightweight validation, documentation checks, and package assembly.

Primary evaluation used the accepted exact-deduplicated strict test set. A conservative dHash-grouped sensitivity run was evaluated as robustness evidence. Metrics included accuracy, 95% confidence interval, macro F1, weighted F1, confusion matrices, expected calibration error, multiclass Brier score, one-vs-rest ROC AUC, and average precision.

### 2.6 Multimodal Vision-Language Model Experiments

Multimodal VLM experiments were run separately from the CNN training pipeline. Qwen and SmolVLM candidates were evaluated with strict JSON prompts. Qwen2.5-VL-3B was also adapted with LoRA. A hierarchical VLM diagnostic prompt was tested by asking for broad tumor-versus-dementia routing before subtype prediction. These experiments were included to determine whether VLMs could serve as direct image classifiers or should remain limited to report and metadata support.

### 2.7 Grounded Report Generation

The application includes a deterministic report generator that converts structured classifier evidence into a draft report. This pathway does not ask an LLM to invent lesion size, lesion location, edema, mass effect, atrophy measurements, or other unsupported findings. The report generator may be described as an application safety feature or moved to an appendix, depending on the target venue.

### 2.8 Statistical And Reproducibility Notes

Accuracy confidence intervals are reported in `docs/PUBLICATION_RESULTS_TABLES.md`. Probability-level evidence is stored under `training_logs/publication_evidence/`. Figure assets and captions are stored under `docs/figures/` and `docs/FIGURE_CAPTIONS.md`. The reproducibility gate is `python3 scripts/check_publication_package.py`.

## 3. Results

### 3.1 Leakage Audit And Corrected Training

The initial audit found 782 exact cross-split SHA-256 overlap rows and 22,537 perceptual dHash overlap rows. The exact duplicates made the initial near-perfect results unsuitable as final claims.

After exact duplicate grouping and regularized retraining, the accepted audit showed 0 exact cross-split overlaps and 0 missing files. The accepted exact-deduplicated split still contained 22,304 dHash overlap rows, so dHash overlap was treated as a sensitivity risk rather than ignored. The corrected dHash-grouped sensitivity run produced 0 exact overlaps, 0 dHash overlaps, and 0 missing files.

### 3.2 Primary Exact-Deduplicated CNN Results

The accepted exact-deduplicated CNN suite achieved strong strict-test performance. The single 8-class CNN was the top integrated classifier, with 0.9972 accuracy, 0.9905 macro F1, and 0.9972 weighted F1. The hierarchical CNN reached 0.9963 accuracy, 0.9891 macro F1, and 0.9963 weighted F1. The binary router reached 1.0000 accuracy, the tumor specialist reached 0.9792 accuracy, and the dementia specialist reached 0.9991 accuracy.

Figure 3 shows the accepted exact-deduplicated confusion matrices. Table 1 should report the primary CNN metrics from `docs/PUBLICATION_RESULTS_TABLES.md`.

### 3.3 Conservative dHash Sensitivity Results

The conservative dHash sensitivity split retained strong performance after grouping exact duplicates and identical dHash fingerprints. The single 8-class CNN reached 0.9906 accuracy, 0.9740 macro F1, and 0.9904 weighted F1. The hierarchical CNN reached 0.9894 accuracy, 0.9755 macro F1, and 0.9893 weighted F1. The tumor specialist reached 0.9539 accuracy, and the dementia specialist reached 0.9975 accuracy.

Figure 4 shows the dHash sensitivity confusion matrices. These results support the accepted exact-deduplicated baseline but should be presented as a conservative robustness analysis, not as the default checkpoint set.

### 3.4 Calibration And Probability-Level Evidence

Probability-level evidence was generated from the accepted checkpoints without retraining. The tumor specialist reached ROC AUC 0.9975 and average precision 0.9875, with expected calibration error 0.0372. The dementia specialist reached ROC AUC 1.0000 and average precision 1.0000, with expected calibration error 0.0386.

The integrated models remained highly discriminative but were less well calibrated. The single 8-class CNN reached ROC AUC 0.9979 and average precision 0.9969, with expected calibration error 0.1327. The hierarchical CNN reached ROC AUC 0.9968 and average precision 0.9907, with expected calibration error 0.1526. Therefore, confidence scores should be described as model confidence, not calibrated clinical probability.

Figures 6 and 7 show calibration, confidence, ROC, and precision-recall evidence.

### 3.5 Multimodal VLM Results

Multimodal VLMs were substantially weaker than CNNs for direct MRI image classification. The best flat zero-shot VLM result was Qwen2.5-VL-7B at 0.1750 accuracy. Qwen2.5-VL-3B with LoRA improved direct labeling to 0.3125 accuracy but showed label collapse. The best hierarchical VLM diagnostic reached 0.8750 broad-domain accuracy but only 0.2250 routed 8-class accuracy.

Figure 5 compares CNN and VLM results. These findings support keeping CNNs as the diagnostic image classifiers and using language models only for constrained report or metadata assistance.

## 4. Discussion

This study shows that strong internal public-dataset performance remains possible after correcting exact duplicate leakage and testing a stricter perceptual-hash sensitivity split. The single 8-class CNN slightly outperformed the hierarchical CNN in both the accepted exact-deduplicated run and the conservative dHash sensitivity run. The hierarchical design remains useful for interpretability because it separates broad-domain routing from domain-specific subtype prediction, but the single-head baseline should be reported as the top strict-test classifier.

The most important methodological lesson is that near-perfect performance should trigger leakage and source-bias audits. In this project, the first full audit found exact duplicate leakage across splits, which changed the publication path. Rather than discarding that finding, the paper should present it as a methodological correction and an example of why public medical-image benchmarks require careful split auditing.

The dHash sensitivity run is also important. Exact duplicate removal is necessary but may not address visually similar augmented images. Grouping by identical dHash fingerprints is conservative and may group clinically distinct but visually similar MRI slices. Performance decreased modestly but remained high, especially for the integrated classifiers, which supports the robustness of the primary result while still warning that patient-level validation is unavailable.

The VLM experiments provide a negative but useful result. Current open multimodal VLMs tested here did not approach CNN performance for direct MRI labeling. Even LoRA adaptation improved only modestly and collapsed some labels. This supports a non-hallucinating design in which deterministic CNN outputs are used for classification and language models, if used at all, are constrained to style or metadata tasks with strict fact preservation.

## 5. Limitations

This study uses public Kaggle image datasets and should not be interpreted as a clinical validation study. Patient identifiers, scanner metadata, acquisition parameters, and institutional source metadata were unavailable. Patient-level leakage cannot be excluded. The dementia dataset is augmented and upsampled before this project receives it, so performance may reflect augmented-image regularities. Tumor and dementia images originate from different dataset sources, so the binary router may learn source-domain artifacts. The dHash sensitivity analysis reduces one near-duplicate risk but does not replace patient-level grouping. Calibration is imperfect for the integrated CNN models. External validation on independent MRI cohorts is required before any clinical or deployment claims.

## 6. Ethics, Data Availability, And Code Availability

This study used publicly available, de-identified image datasets from Kaggle. No patient-level identifiers, acquisition metadata, or institutional clinical records were available to the investigators. Because only public de-identified datasets were used, institutional review board review and informed consent were not sought for this computational analysis. This statement should be adjusted if the target venue requires formal institutional review wording.

The raw images are not redistributed in this repository. They can be obtained from the cited Kaggle dataset pages subject to their current terms and licenses. Derived manifests, evaluation metrics, confusion matrices, calibration summaries, probability outputs, and manuscript figures are included in this repository.

Code, documentation, trained model artifacts, and publication evidence are maintained in the GitHub repository `TailsOS-hack/Tumor-Database`.

## 7. Conclusion

After leakage correction and robustness testing, CNN classifiers achieved strong internal strict-test performance for combined brain tumor and dementia MRI image classification. The single 8-class CNN was the highest-performing integrated architecture, while the hierarchical pipeline provides interpretable routing and specialist analysis. Multimodal VLMs were not competitive as direct image classifiers in these experiments. The work is suitable as a leakage-audited public-dataset benchmark and application-safety study, but it requires external validation and patient-level metadata before clinical claims.

## Figure Callouts

- Figure 1: Dataset splitting and leakage-audit workflow.
- Figure 2: Hierarchical versus single-head architecture comparison.
- Figure 3: Accepted exact-deduplicated confusion matrices.
- Figure 4: Conservative dHash sensitivity confusion matrices.
- Figure 5: CNN versus VLM comparison.
- Figure 6: Calibration and confidence evidence.
- Figure 7: ROC and precision-recall evidence.

## Table Callouts

- Table 1: Accepted exact-deduplicated CNN strict-test metrics.
- Table 2: Conservative dHash sensitivity strict-test metrics.
- Table 3: Leakage and robustness audit checks.
- Table 4: Multimodal VLM benchmark results.
- Table 5: Probability-level evidence and calibration metrics.

## Reference Placeholders

Replace this section with the target venue's required reference style.

1. Nickparvar M. Brain Tumor MRI Dataset. Kaggle; 2021. DOI: 10.34740/KAGGLE/DSV/2645886.
2. Singhal A. Alzheimer's Disease Multiclass Images Dataset. Kaggle.
3. UraninJo. Augmented Alzheimer MRI Dataset V2. Kaggle.
4. Dubey S / Tourist55. Alzheimer's Dataset (4 class of Images). Kaggle.
5. Add ResNet reference.
6. Add EfficientNet reference.
7. Add MobileNetV3 reference.
8. Add Qwen/Qwen2.5-VL references if VLM experiments remain in the main manuscript.
9. Add LoRA reference if LoRA experiments remain in the main manuscript.
10. Add calibration/ECE reference if the target venue expects methodological citations.

## Final Editing Notes

- Choose a target venue before final formatting.
- Replace all placeholder author, affiliation, funding, conflict-of-interest, and reference fields.
- Confirm live Kaggle license fields before submission.
- Decide whether the deterministic report generator belongs in the main paper, appendix, or separate application note.
- Keep the calibration limitation visible in the abstract, results, and limitations.
