# Kaggle Multimodal Results

This records the free Kaggle multimodal batches for the tumor/dementia project.

## Current Takeaway

The CNN classifiers remain the publication-grade image classifiers. Multimodal VLMs can produce structured JSON, but they are not competitive as direct MRI image classifiers in the current setup. Batch 2 LoRA improved label accuracy from roughly 17% zero-shot to 31.25%, but the adapter still collapsed many examples into a few labels.

## Batch 1

- Kernel: `armankazi/tumor-multimodal-qwen-kaggle`
- Run date: 2026-05-11
- GPU: two Tesla T4 GPUs, 29.12 GB aggregate memory
- Strict manifest rows: 51,023
- Local artifact path: `training_logs/multimodal/kaggle_qwen_batch1/`
- LoRA adapter path: `models/multimodal/qwen25vl_3b_mri_lora/` was updated by batch 2; batch 1 adapter is retained in Git history.

## Zero-Shot Benchmark

| Candidate | Status | Test Sample | Strict JSON Rate | Label Accuracy | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-3B-Instruct | completed | 64 | 1.0000 | 0.1719 | Best zero-shot result, but not competitive with CNN classifiers |
| Qwen/Qwen2-VL-2B-Instruct | completed | 48 | 1.0000 | 0.1250 | Smaller Qwen baseline |
| Qwen/Qwen2.5-VL-7B-Instruct | completed | 40 | 1.0000 | 0.1500 | Fit on two T4 GPUs in 4-bit |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | failed | 0 | N/A | N/A | Missing `num2words` dependency in batch 1 |
| llava-hf/llava-v1.6-34b-hf | skipped | 0 | N/A | N/A | Needs more than the available 29.12 GB aggregate T4 memory |

## LoRA Adapter

| Field | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-VL-3B-Instruct |
| Training examples | 64 |
| Validation examples | 16 |
| Optimizer updates | 16 |
| Final loss | 0.050605 |
| Mean loss | 0.376196 |
| Adapter | `models/multimodal/qwen25vl_3b_mri_lora/adapter_model.safetensors` |

## Batch 2

- Kernel: `armankazi/tumor-multimodal-qwen-kaggle`
- Kernel version: 4
- Run date: 2026-05-11
- GPU: two Tesla T4 GPUs, 29.12 GB aggregate memory
- Strict manifest rows: 51,023
- Local artifact path: `training_logs/multimodal/kaggle_qwen_batch2/`
- Current LoRA adapter path: `models/multimodal/qwen25vl_3b_mri_lora/`

## Batch 2 Zero-Shot Benchmark

| Candidate | Status | Test Sample | Strict JSON Rate | Label Accuracy | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-3B-Instruct | completed | 64 | 1.0000 | 0.1719 | Collapsed mostly to `tumor_notumor` |
| Qwen/Qwen2-VL-2B-Instruct | completed | 48 | 1.0000 | 0.1250 | Smaller Qwen baseline |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | completed | 48 | 0.7292 | 0.0833 | Dependency fixed, but JSON and accuracy were weak |
| Qwen/Qwen2.5-VL-7B-Instruct | completed | 40 | 1.0000 | 0.1750 | Best zero-shot result in batch 2 |
| llava-hf/llava-v1.6-34b-hf | skipped | 0 | N/A | N/A | Needs more than the available 29.12 GB aggregate T4 memory |

## Batch 2 LoRA Adapter

| Field | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-VL-3B-Instruct |
| Training examples | 256 |
| Validation examples | 64 |
| Optimizer updates | 64 |
| Final loss | 0.036855 |
| Mean loss | 0.123053 |
| Adapter | `models/multimodal/qwen25vl_3b_mri_lora/adapter_model.safetensors` |

## Batch 2 LoRA Evaluation

| Model | Test Sample | Strict JSON Rate | Label Accuracy | Main Failure Mode |
| --- | ---: | ---: | ---: | --- |
| Qwen/Qwen2.5-VL-3B-Instruct + LoRA | 96 | 1.0000 | 0.3125 | Tumor labels partly separated, dementia subclasses collapsed to `dementia_ModerateDemented` |

Per-domain LoRA accuracy on the balanced 96-image sample:

| Domain | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Tumor | 18 | 48 | 0.3750 |
| Dementia | 12 | 48 | 0.2500 |

## Interpretation

The multimodal VLMs produced strict JSON reliably, but their MRI label accuracy was weak. Batch 2 LoRA improved over zero-shot prompting but still showed label collapse. This supports keeping the CNN classifier suite as the primary image classifier and treating multimodal models as experimental report/metadata helpers unless the task is redesigned.

## Batch 3 Configuration

Batch 3 is a targeted hierarchical diagnostic instead of another flat 8-class LoRA run.

- Kernel: `armankazi/tumor-multimodal-qwen-kaggle`
- Script: `notebooks/kaggle_multimodal_qwen_kernel.py`
- Models: `Qwen/Qwen2.5-VL-3B-Instruct` and, if enough GPU memory is assigned, `Qwen/Qwen2.5-VL-7B-Instruct`
- Method: ask tumor-vs-dementia first, then ask the matching 4-way subtype prompt
- Metrics: domain accuracy, routed hierarchical 8-class accuracy, oracle-domain subtype accuracy, JSON rates, confusion matrices, prediction counts
- LoRA: disabled for this batch so the run isolates prompt/task structure from adapter training

The recommended next multimodal direction is not another blind 8-class VLM classifier run. Instead, use the trained CNN image classifiers for diagnosis labels and evaluate the VLM as a report explainer or metadata generator conditioned on classifier probabilities, or continue redesigning multimodal training into smaller hierarchical tasks.
