# Publication Notes Template

Use this file as the running methods/results record. Fill it only from strict test outputs produced by `src.experiment_pipeline`.

## Study Goal

Compare a hierarchical MRI classifier against a single 8-class classifier for tumor and dementia image categorization, then evaluate whether multimodal LLMs improve report metadata extraction or report generation.

## Dataset and Splits

| Dataset | Classes | Split Source | Notes |
| --- | --- | --- | --- |
| Brain tumor | glioma, meningioma, notumor, pituitary | Official Training/Testing folders | Testing folder held out as strict test |
| Dementia | MildDemented, ModerateDemented, NonDemented, VeryMildDemented | Deterministic stratified split | Manifest generated before augmentation |

Manifest path: `training_logs/splits/strict_manifest.csv`

## Architectures

| Experiment | Router | Specialist / Head | Output Space | Checkpoint |
| --- | --- | --- | --- | --- |
| Hierarchical | ResNet50 binary router | EfficientNet-B3 tumor, MobileNetV3 dementia | Binary route plus 4-class specialists | `models/binary_router.pt`, specialists |
| Single model | None | EfficientNet-B3 | 8 classes | `models/single_8class_classifier.pt` |
| Multimodal LLM | Qwen/Qwen2.5-VL-3B-Instruct | LoRA adapter | Structured JSON labels/report metadata | `models/multimodal/qwen25vl_3b_mri_lora/` |

## Strict-Test Results

| Model | Accuracy | Macro F1 | Weighted F1 | Key Failure Modes | Metrics Path |
| --- | ---: | ---: | ---: | --- | --- |
| Binary router | 1.0000 | 1.0000 | 1.0000 | Possible domain/source bias should be discussed | `training_logs/experiments/binary/20260510_204656/test/metrics.json` |
| Tumor specialist | 0.9962 | 0.9960 | 0.9962 | Rare subtype confusions | `training_logs/experiments/tumor/20260510_222255/test/metrics.json` |
| Dementia specialist | 0.9985 | 0.9986 | 0.9985 | Dementia split may be optimistic because no official holdout was provided | `training_logs/experiments/dementia/20260510_224400/test/metrics.json` |
| Hierarchical end-to-end | 0.9982 | 0.9973 | 0.9982 | Inherits router and specialist risks | `training_logs/experiments/hierarchical/test_evaluation/metrics.json` |
| Single 8-class | 0.9993 | 0.9975 | 0.9993 | Strongest strict-test accuracy but still needs leakage discussion | `training_logs/experiments/eight_class/20260511_000016/test/metrics.json` |

## Multimodal LLM Benchmark

| Candidate | Size | Runtime | Quantization | Strict JSON Rate | Label Accuracy | Report Quality Notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1719 | Best zero-shot VLM, still weak |
| Qwen/Qwen2-VL-2B-Instruct | 2B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1250 | Smaller Qwen baseline |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | Kaggle 2x T4 | 4-bit | 1.0000 | 0.1750 | Best zero-shot in batch 2, still weak |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | Kaggle 2x T4 | 4-bit | 0.7292 | 0.0833 | Dependency fixed in batch 2; JSON and accuracy weak |
| Qwen/Qwen2.5-VL-3B-Instruct + LoRA | 3B | Kaggle 2x T4 | 4-bit LoRA | 1.0000 | 0.3125 | Improved after 256-example LoRA but still collapsed labels |
| llava-hf/llava-v1.6-34b-hf | 34B | Kaggle 2x T4 | 4-bit | N/A | N/A | Skipped because memory was insufficient |

## Methods Notes

- Report all final metrics from the strict test split.
- Include confusion matrices for every classifier and end-to-end hierarchical inference.
- Keep validation metrics separate from test metrics.
- Record random seed, epochs, image size, optimizer, learning rate, batch size, GPU type, and checkpoint hash for each run.
- For LoRA, report base model, adapter rank, target modules, quantization, training examples, validation examples, and adapter path.
- Current multimodal conclusion: VLMs should not replace the CNN classifiers for image labeling; treat them as report/metadata assistants unless a redesigned task produces materially stronger strict-test accuracy.

## Rationale Draft

The hierarchical approach is expected to reduce subtype confusion by first separating broad image domains, then applying specialized classifiers trained on narrower label spaces. The 8-class baseline tests whether a single shared visual representation can learn both datasets without routing errors. The multimodal LLM experiments are treated separately because they are evaluated for structured metadata/report generation and should not replace strict visual classifier metrics without evidence.
