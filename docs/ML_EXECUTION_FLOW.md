# ML Execution Flow

This is the clean restart plan for the tumor/dementia project. Training and large multimodal work should run in Colab or a remote runner, not on the MacBook.

## Order of Operations

1. Create strict splits before augmentation.

   ```bash
   python -m src.experiment_pipeline create-manifest
   python -m src.experiment_pipeline summary
   ```

2. Smoke-check the training code locally with tiny, non-pretrained runs only.

   ```bash
   python -m src.experiment_pipeline train --task binary --epochs 1 --smoke-test --no-pretrained
   ```

3. Run the classifier suite in the cloud.

   The GitHub Actions workflow `.github/workflows/ml-experiment-suite.yml` runs the manifest, binary router, tumor specialist, dementia specialist, hierarchical evaluation, single 8-class baseline, and publication-summary collection. Use `smoke` on `ubuntu-latest`; use `full` on a CUDA/GPU runner.

   For a one-shot Google Colab run that trains and then downloads a single artifact zip, use `docs/COLAB_FULL_TRAINING.md`.

   Recommended full-run dispatch settings:

   - `mode`: `full`
   - `runner`: `["self-hosted","linux","gpu"]` or the label JSON for the configured GPU runner
   - `epochs`: `30`
   - `batch_size`: `32`
   - `allow_cpu_full`: `false`

4. Train the binary router manually if running outside the workflow.

   ```bash
   python -m src.experiment_pipeline train --task binary --epochs 30 --batch-size 32 --pretrained
   ```

5. Validate the binary router on the strict test split.

   ```bash
   python -m src.experiment_pipeline test --task binary --split test
   ```

6. Retrain the tumor and dementia specialists using the same strict manifest and global training augmentation.

   ```bash
   python -m src.experiment_pipeline train --task tumor --epochs 30 --batch-size 32 --pretrained
   python -m src.experiment_pipeline train --task dementia --epochs 30 --batch-size 32 --pretrained
   ```

7. Test end-to-end hierarchical accuracy.

   ```bash
   python -m src.experiment_pipeline evaluate-hierarchical --split test
   ```

8. Run the parallel single 8-class model experiment.

   ```bash
   python -m src.experiment_pipeline train --task eight_class --epochs 30 --batch-size 32 --pretrained
   python -m src.experiment_pipeline test --task eight_class --split test
   ```

9. Move multimodal LLM inference to cloud GPU and test 3-5 models.

   Use `notebooks/multimodal_llm_lora_colab.ipynb` for Colab or `notebooks/kaggle_multimodal_qwen_kernel.py` for Kaggle. The current free-runner path is Kaggle with a T4 accelerator request and hourly monitoring from Codex.

10. Fine-tune the best multimodal model with LoRA.

   Start from the highest validation accuracy/model-quality candidate in the Colab benchmark and train adapters only. Keep the base model frozen and save adapter weights to Drive.

11. Document results for publication.

   Run `python scripts/collect_publication_results.py` after the suite, then fill `docs/PUBLICATION_NOTES.md`. Do not report validation-only numbers as final results.

12. Audit the CNN results for leakage and overfitting before manuscript submission.

   Local summary-only audit:

   ```bash
   python scripts/publication_audit.py --skip-image-hashes --output-dir training_logs/publication_audit/local_summary
   ```

   Full remote audit and regularized retraining:

   ```bash
   kaggle kernels push -p <clean-upload-folder> --accelerator NvidiaTeslaT4
   ```

   The Kaggle script is `notebooks/kaggle_cnn_publication_audit_kernel.py`. It performs image-hash duplicate checks, current checkpoint train/val/test evaluation, and a regularized retraining suite. The strict manifest builder now groups exact duplicate SHA-256 image hashes into one split by default. Use `--allow-duplicate-leakage` only for legacy reproduction, never for publication training.

## Leakage Controls

- The manifest is created before augmentation.
- Training transforms include horizontal flips, rotations, and contrast jitter.
- Validation and test transforms only resize, tensorize, and normalize.
- Tumor images in `data/brain_tumor/Testing` are always held out as test images.
- Dementia images are stratified into deterministic train/val/test splits because this repo does not ship an official dementia test folder.
- Every model task uses the same manifest so the hierarchical and 8-class comparisons are aligned.

## Action Tracker

| Action Item | Description | Assigned To | Status |
| --- | --- | --- | --- |
| Build binary classifier | Tumor vs dementia, labels `tumor=0`, `dementia=1` | Arman | Automated in GitHub Actions full suite |
| Apply image augmentation | Flips, rotations, contrast on all training datasets | Arman | Implemented in shared training transforms |
| Retrain subtype models | Tumor and dementia specialists with strict splits | Arman | Automated in GitHub Actions full suite |
| Validate confusion matrices | Save realistic strict-test metrics and confusion matrices | Arman | Automated: metrics JSON, CSV, PNG outputs |
| Set up cloud GPU runner | Kaggle free GPU first, Colab as fallback | Arman | Kaggle CLI runner active |
| Test large multimodal models | LLaVA-34B, Qwen, Phi, Llama/other candidates | Arman | Batch 1, 2, and 3 complete |
| Implement LoRA fine-tuning | Adapter-based training | Arman | Batch 2 Qwen2.5-VL-3B adapter saved |
| Compare architectures | Binary+specialist vs single 8-class | Arman | Automated in GitHub Actions full suite |
| Prepare publication notes | Model comparisons and rationale | Arman / Mina | Results plus audit workflow added |
| Audit overfitting/leakage | Hash overlap checks, train/val/test gaps, regularized CNN rerun | Arman | Complete: exact duplicate leakage fixed; de-duplicated checkpoints accepted |

## Output Locations

- Split manifest: `training_logs/splits/strict_manifest.csv`
- Model checkpoints: `models/binary_router.pt`, `models/brain_tumor_classifier.pt`, `models/alzheimers_classifier.pt`, `models/single_8class_classifier.pt`
- Metrics: `training_logs/experiments/<task>/<run>/test/`
- End-to-end metrics: `training_logs/experiments/hierarchical/test_evaluation/`
- Colab export bundle: `MyDrive/Tumor-Database/exports/tumor_database_colab_artifacts_*.zip`
- Kaggle multimodal batch 1: `training_logs/multimodal/kaggle_qwen_batch1/`
- Kaggle multimodal batch 2: `training_logs/multimodal/kaggle_qwen_batch2/`
- Kaggle multimodal batch 3: `training_logs/multimodal/kaggle_qwen_batch3/` after artifact import
- Local publication audit: `training_logs/publication_audit/local_summary/`
- Initial Kaggle CNN audit output: `training_logs/publication_audit/regularized/`, `training_logs/experiments_regularized/`
- Accepted de-duplicated CNN audit output: `training_logs/publication_audit/dedup_regularized/`, `training_logs/experiments_dedup_regularized/`
- Current Kaggle LoRA adapter: `models/multimodal/qwen25vl_3b_mri_lora/`

## Multimodal Candidate References

- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
- [llava-hf/llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)
- [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
