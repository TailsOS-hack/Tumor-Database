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
| Multimodal LLM | Candidate-dependent | Prompting or LoRA adapter | Structured report metadata/labels | Google Drive adapter path |

## Strict-Test Results

| Model | Accuracy | Macro F1 | Weighted F1 | Key Failure Modes | Metrics Path |
| --- | ---: | ---: | ---: | --- | --- |
| Binary router | TBD | TBD | TBD | TBD | `training_logs/experiments/binary/.../test/metrics.json` |
| Tumor specialist | TBD | TBD | TBD | TBD | `training_logs/experiments/tumor/.../test/metrics.json` |
| Dementia specialist | TBD | TBD | TBD | TBD | `training_logs/experiments/dementia/.../test/metrics.json` |
| Hierarchical end-to-end | TBD | TBD | TBD | TBD | `training_logs/experiments/hierarchical/test_evaluation/metrics.json` |
| Single 8-class | TBD | TBD | TBD | TBD | `training_logs/experiments/eight_class/.../test/metrics.json` |

## Multimodal LLM Benchmark

| Candidate | Size | Runtime | Quantization | Strict JSON Rate | Label Accuracy | Report Quality Notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | Colab A100 | 4-bit or bf16 | TBD | TBD | TBD |
| Qwen/Qwen2.5-VL-32B-Instruct | 32B | Colab A100 | 4-bit | TBD | TBD | TBD |
| llava-hf/llava-v1.6-34b-hf | 34B | Colab A100 | 4-bit | TBD | TBD | TBD |
| microsoft/Phi-3.5-vision-instruct | 4.2B | Colab A100/T4 | bf16 or 4-bit | TBD | TBD | TBD |
| meta-llama/Llama-3.2-11B-Vision-Instruct | 11B | Colab A100 | 4-bit or bf16 | TBD | TBD | Gated model access required |

## Methods Notes

- Report all final metrics from the strict test split.
- Include confusion matrices for every classifier and end-to-end hierarchical inference.
- Keep validation metrics separate from test metrics.
- Record random seed, epochs, image size, optimizer, learning rate, batch size, GPU type, and checkpoint hash for each run.
- For LoRA, report base model, adapter rank, target modules, quantization, training examples, validation examples, and adapter path.

## Rationale Draft

The hierarchical approach is expected to reduce subtype confusion by first separating broad image domains, then applying specialized classifiers trained on narrower label spaces. The 8-class baseline tests whether a single shared visual representation can learn both datasets without routing errors. The multimodal LLM experiments are treated separately because they are evaluated for structured metadata/report generation and should not replace strict visual classifier metrics without evidence.
