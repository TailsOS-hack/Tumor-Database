# Dataset Provenance, Attribution, And License Notes

This document records the dataset sources used by the current publication package. It is intended for manuscript drafting and reviewer traceability. Re-check the Kaggle data cards immediately before final submission because dataset owners can update descriptions, versions, and license metadata.

Access date for this provenance pass: 2026-05-14.

## Local Dataset Layout

The strict manifests reference two local dataset roots:

| Local root | Domain | Classes | Manifest source split behavior |
| --- | --- | --- | --- |
| `data/brain_tumor/` | Brain tumor MRI | `glioma`, `meningioma`, `notumor`, `pituitary` | Official `Training/` images form the train/validation pool; official `Testing/` images are preferentially held out as test images. |
| `data/alzheimers/` | Dementia MRI | `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented` | No official local test folder is present; images are deterministically stratified into train/validation/test after duplicate grouping. |

## Primary Dataset Sources

| Domain | Dataset to cite | URL / DOI | Local evidence match | License notes |
| --- | --- | --- | --- | --- |
| Brain tumor MRI | Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle | `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`; DOI `https://doi.org/10.34740/KAGGLE/DSV/2645886` | Local class names and folder layout match the Kaggle dataset: `Training/` and `Testing/` folders with glioma, meningioma, no-tumor, and pituitary classes. The manifest contains 5,712 training-pool rows and 1,311 official test rows, matching the 7,023-image dataset commonly reported for this source. | Current attribution should cite the Kaggle dataset and DOI. The dataset license is commonly reported as CC0-1.0 / public domain in downstream dataset metadata, but re-check the Kaggle data card before final submission. |
| Dementia MRI | Aryan Singhal, *Alzheimer's Disease Multiclass Images Dataset*, Kaggle | `https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented` | Local counts match the data card: 10,000 MildDemented, 10,000 ModerateDemented, 12,800 NonDemented, and 11,200 VeryMildDemented images, 44,000 total. | Kaggle data card lists Apache 2.0 for this dataset. The data card states that it is an augmented/upsampled version of UraninJo's augmented Alzheimer MRI dataset, so cite the derivative source chain and re-check license compatibility before submission. |

## Dementia Upstream Source Chain

The dementia dataset should be described as a Kaggle derivative dataset, not as a clinically curated patient cohort.

| Role | Dataset | URL | Notes |
| --- | --- | --- | --- |
| Dataset used locally | Aryan Singhal, *Alzheimer's Disease Multiclass Images Dataset* | `https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented` | Approx. 44,000 skull-stripped JPG MRI images; class counts match the local manifest. |
| Declared upstream | UraninJo, *Augmented Alzheimer MRI Dataset V2* | `https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2` | Data card states that it contains originals and augmented images derived from the Tourist55 Alzheimer's dataset; license listed as GNU LGPL 3.0. |
| Original upstream referenced by UraninJo | Tourist55, *Alzheimer's Dataset (4 class of Images)* | `https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images` | Commonly cited in Alzheimer MRI image-classification papers; reported image count and license metadata vary across mirrors and publications, so verify directly before final submission. |

## Suggested Manuscript Wording

Dataset paragraph:

> Brain tumor MRI images were sourced from the public Kaggle Brain Tumor MRI Dataset by Masoud Nickparvar, which contains glioma, meningioma, no-tumor, and pituitary MRI classes with official Training and Testing folders. Dementia MRI images were sourced from the public Kaggle Alzheimer's Disease Multiclass Images Dataset by Aryan Singhal, an augmented and upsampled derivative dataset organized into MildDemented, ModerateDemented, NonDemented, and VeryMildDemented classes. The dementia dataset data card cites UraninJo's Augmented Alzheimer MRI Dataset V2 as its upstream source, which in turn references Tourist55's Alzheimer's Dataset (4 class of Images).

Ethics and consent paragraph:

> This study used publicly available, de-identified image datasets from Kaggle. No patient-level identifiers, acquisition metadata, or institutional clinical records were available to the investigators. Because only public de-identified datasets were used, institutional review board review and informed consent were not sought for this computational analysis. The absence of patient-level metadata prevents patient-level split validation and is reported as a limitation.

Data availability paragraph:

> The raw images are not redistributed in this repository. They can be obtained from the cited Kaggle dataset pages subject to their current terms and licenses. Derived manifests, evaluation metrics, confusion matrices, calibration summaries, probability outputs, and manuscript figures are included in this repository to support reproducibility of the reported results.

License paragraph:

> The brain tumor and dementia datasets are reused under the licenses displayed by their respective Kaggle data cards at the time of access. Before publication, the authors should re-check the current Kaggle license fields and retain a dated copy or screenshot of each data card for the project record.

## Citation Drafts

Use the target venue's citation style, but keep these source details:

- Nickparvar M. *Brain Tumor MRI Dataset*. Kaggle; 2021. DOI: `10.34740/KAGGLE/DSV/2645886`. Available at: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`.
- Singhal A. *Alzheimer's Disease Multiclass Images Dataset*. Kaggle. Available at: `https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented`.
- UraninJo. *Augmented Alzheimer MRI Dataset V2*. Kaggle. Available at: `https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2`.
- Dubey S / Tourist55. *Alzheimer's Dataset (4 class of Images)*. Kaggle. Available at: `https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images`.

## Publication Caveats

- The study should not claim patient-level independence because patient identifiers are unavailable.
- The dementia dataset is augmented and upsampled before this project receives it; reported performance may reflect augmented-image regularities.
- Tumor and dementia images originate from different dataset sources, so the binary router may learn source/domain artifacts.
- License metadata must be checked again at final submission, especially for the dementia derivative source chain.
- Do not redistribute raw Kaggle images in the repository unless the final license review explicitly allows it and attribution requirements are satisfied.
