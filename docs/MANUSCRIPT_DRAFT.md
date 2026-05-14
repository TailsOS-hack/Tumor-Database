# Manuscript Draft

## Working Title

Leakage-Audited Hierarchical and Single-Head CNNs for Brain MRI Tumor and Dementia Image Classification

## Abstract Draft

This study evaluates convolutional neural network classifiers for brain MRI image categorization across tumor and dementia datasets. We compare a hierarchical pipeline, consisting of a binary tumor-versus-dementia router followed by domain-specific subtype classifiers, against a single 8-class CNN baseline. Because initial strict-test performance was near-perfect, we performed image-hash leakage audits before defining publishable results. The first full audit identified exact cross-split duplicate leakage, so final model claims use a de-duplicated retraining run that assigns exact duplicate SHA-256 image groups to a single split before augmentation. The accepted exact-deduplicated CNN suite reached 0.9963 hierarchical accuracy and 0.9972 single 8-class accuracy on the strict test split. A conservative perceptual-hash sensitivity run, which also grouped identical dHash fingerprints into one split, retained strong performance with 0.9894 hierarchical accuracy and 0.9906 single 8-class accuracy. Multimodal vision-language model experiments, including zero-shot, hierarchical prompting, and LoRA adaptation, were substantially weaker for direct image classification. These results support CNNs as the primary image classifiers for this dataset, while emphasizing source-bias limitations and the need for external validation.

## Methods Draft

Images were treated as brain MRI scans. The brain tumor dataset contains glioma, meningioma, no-tumor, and pituitary classes. The dementia dataset contains MildDemented, ModerateDemented, NonDemented, and VeryMildDemented classes. Splits were generated before augmentation. Tumor images from the official Testing folder were preferentially held out as test data, while dementia images were deterministically stratified because no official dementia test split was available.

The primary training manifest grouped exact duplicate SHA-256 image hashes into a single split to prevent identical pixels from appearing across train, validation, and test partitions. Training augmentation was applied only to training images and included horizontal flips, rotation, and contrast jitter. Validation and test transforms were limited to resizing, tensor conversion, and normalization.

The hierarchical architecture used a ResNet50 binary router to classify images as tumor or dementia, followed by an EfficientNet-B3 tumor specialist or MobileNetV3 dementia specialist. The comparison baseline used a single EfficientNet-B3 head over all eight classes. Training used ImageNet-pretrained backbones, class-weighted cross-entropy, label smoothing, random erasing, AdamW optimization, stronger weight decay, and early stopping.

Publication readiness was assessed with duplicate-path checks, exact SHA-256 cross-split overlap checks, perceptual dHash cross-split overlap checks, train-validation gap summaries, strict-test metrics, and confusion matrices. A corrected perceptual-hash sensitivity run was performed after aligning the manifest-builder dHash implementation with the audit implementation.

Multimodal vision-language model experiments were run on Kaggle using Qwen and SmolVLM candidates with strict JSON prompts, hierarchical prompts, and Qwen2.5-VL-3B LoRA adaptation. These models were evaluated separately from the CNN classifiers.

## Results Draft

The initial full audit found 782 exact cross-split SHA-256 overlap rows and was therefore treated as a blocking leakage risk. After exact-duplicate grouping and retraining, exact cross-split overlaps were reduced to 0, missing files remained 0, and no train-validation overfitting gap flags were found. The accepted exact-deduplicated CNN run achieved:

- Binary router: 1.0000 accuracy.
- Tumor specialist: 0.9792 accuracy.
- Dementia specialist: 0.9991 accuracy.
- Hierarchical CNN: 0.9963 accuracy.
- Single 8-class CNN: 0.9972 accuracy.

The exact-deduplicated audit still reported identical perceptual dHash values across splits. Because dHash is coarse for MRI slices and can group visually similar but non-identical images, this was treated as a sensitivity question rather than an automatic blocker. A corrected dHash-grouped sensitivity run produced 0 exact overlaps, 0 perceptual dHash overlaps, 0 missing files, and 0 train-validation gap flags. Under that stricter split, performance remained high:

- Binary router: 1.0000 accuracy.
- Tumor specialist: 0.9539 accuracy.
- Dementia specialist: 0.9975 accuracy.
- Hierarchical CNN: 0.9894 accuracy.
- Single 8-class CNN: 0.9906 accuracy.

The single 8-class CNN slightly outperformed the hierarchical CNN in both the accepted exact-deduplicated run and the conservative dHash sensitivity run. The hierarchical design remains useful for interpretability and domain-specific error analysis, but the single-head baseline should be reported as the top strict-test performer.

Multimodal VLMs were not competitive as direct MRI classifiers. The best flat zero-shot VLM result was Qwen2.5-VL-7B at 0.1750 accuracy. Qwen2.5-VL-3B with LoRA improved to 0.3125 accuracy but showed label collapse. The best hierarchical VLM diagnostic reached 0.8750 broad-domain accuracy but only 0.2250 routed 8-class accuracy.

## Tables And Figures

Primary generated tables:

- `docs/PUBLICATION_RESULTS_TABLES.md`
- `docs/publication_cnn_results.csv`
- `docs/publication_audit_checks.csv`
- `docs/publication_vlm_results.csv`
- `docs/PUBLICATION_FIGURES.md`

Recommended figures:

- Figure 1: Dataset and splitting workflow, including exact-hash and dHash audit stages.
- Figure 2: Model architecture comparison: hierarchical router plus specialists versus single 8-class CNN.
- Figure 3: Accepted exact-deduplicated confusion matrices for hierarchical and single 8-class CNNs.
- Figure 4: Conservative dHash sensitivity confusion matrices for hierarchical and single 8-class CNNs.
- Figure 5: Multimodal VLM comparison showing direct VLM classification failure relative to CNNs.

## Limitations

The binary router may exploit dataset/source differences because tumor and dementia images originate from separate source datasets. Dementia results may be optimistic because no official external dementia holdout split was available. The dHash sensitivity run reduces near-duplicate split risk, but dHash is still a coarse perceptual fingerprint rather than a patient-level grouping method. The study should not make clinical deployment claims without independent external MRI validation and patient-level metadata.

## Submission Checklist

- Confirm all final tables are regenerated with `python3 scripts/build_publication_tables.py`.
- Use exact-deduplicated CNN results as the primary table.
- Include the dHash sensitivity table as robustness evidence.
- Include the initial leakage audit as a methodological correction, not as a final result.
- Report VLM experiments as negative direct-classification results.
- Avoid clinical claims beyond this dataset.
