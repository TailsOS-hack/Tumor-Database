# Kaggle Multimodal Results

This records the first free Kaggle multimodal batch for the tumor/dementia project.

## Batch 1

- Kernel: `armankazi/tumor-multimodal-qwen-kaggle`
- Run date: 2026-05-11
- GPU: two Tesla T4 GPUs, 29.12 GB aggregate memory
- Strict manifest rows: 51,023
- Local artifact path: `training_logs/multimodal/kaggle_qwen_batch1/`
- LoRA adapter path: `models/multimodal/qwen25vl_3b_mri_lora/`

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

## Interpretation

The multimodal VLMs produced strict JSON reliably, but their zero-shot MRI label accuracy was weak. This supports keeping the CNN classifier suite as the primary image classifier and treating multimodal models as experimental report/metadata helpers unless fine-tuned evaluation improves substantially.

Batch 2 should evaluate the LoRA adapter directly, increase LoRA training coverage, and retry SmolVLM with its missing dependency installed.
