# Colab Full Training Export

Use this flow when the MacBook should not do the heavy training work.

## One Colab Cell

Open a fresh Google Colab notebook with a GPU runtime, then run:

```python
import subprocess
import sys
import urllib.request

script = "/content/colab_full_training_export.py"
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/TailsOS-hack/Tumor-Database/main/notebooks/colab_full_training_export.py",
    script,
)
result = subprocess.run([sys.executable, script, "--epochs", "30", "--batch-size", "32"])
if result.returncode:
    raise SystemExit(
        "Training failed. Scroll up to the FAILED banner, or send back the downloaded failure zip."
    )
```

What it does:

1. Mounts Google Drive.
2. Fresh-clones `TailsOS-hack/Tumor-Database`.
3. Installs only the training dependencies needed in Colab.
4. Runs the full strict suite:
   - binary tumor-vs-dementia router
   - tumor specialist
   - dementia specialist
   - hierarchical test evaluation
   - single 8-class baseline
   - publication summary collection
5. Packages the model files and metrics into one zip.
6. Copies the zip to `MyDrive/Tumor-Database/exports/`.
7. Starts a browser download of the same zip.
8. Deletes the temporary Colab repo clone after packaging.

## Progress Updates

The Colab output prints large `START`, `DONE`, and `FAILED` banners for each major stage:

- mount Drive
- clone repo
- install dependencies
- check GPU
- run strict training suite
- package models and metrics
- start browser download
- delete temporary clone

Inside the training stage, each model task prints its own command and epoch progress. The export zip also includes `colab_run_progress.jsonl` and `colab_console.log` with the progress events and command output that were recorded before packaging.

If the cell ends with `CalledProcessError`, scroll up to the nearest `FAILED` banner. The runner will also try to download a `tumor_database_colab_failure_*.zip` file that contains `colab_failure.json`, `colab_run_progress.jsonl`, and `colab_console.log`.

The exported zip should include:

- `models/binary_router.pt`
- `models/brain_tumor_classifier.pt`
- `models/alzheimers_classifier.pt`
- `models/single_8class_classifier.pt`
- `training_logs/splits/strict_manifest.csv`
- `training_logs/experiments/**`
- `colab_export_manifest.json`
- `colab_run_progress.jsonl`
- `colab_console.log`

## Local Import After Download

After the zip appears in Downloads, send the file path here. The local import command is:

```bash
python3 scripts/import_colab_training_export.py ~/Downloads/tumor_database_colab_artifacts_YYYYMMDD_HHMMSS.zip --delete-archive
```

The importer:

1. Verifies the export is a full Colab run.
2. Sets up Git LFS tracking for `models/*.pt`.
3. Copies the newest model files into `models/`.
4. Copies strict split and experiment logs locally for verification.
5. Copies publication summaries and the export manifest into `docs/`.
6. Deletes its temporary extraction directory.
7. Deletes the downloaded zip when `--delete-archive` is used.

After import, commit and push the model update with Git LFS. Once GitHub has the final commit, delete any remaining local archive, temporary extraction files, and any extra Colab clone/download folders.
