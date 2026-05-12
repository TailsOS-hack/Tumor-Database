"""Strict training and evaluation pipeline for tumor/dementia experiments.

This module is intentionally runnable as a CLI so heavy training can happen on
Colab or a remote runner while the laptop only prepares manifests and smoke
checks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "training_logs"
SPLITS_DIR = LOGS_DIR / "splits"
EXPERIMENTS_DIR = LOGS_DIR / "experiments"
DEFAULT_MANIFEST = SPLITS_DIR / "strict_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DOMAIN_CLASSES = ["tumor", "dementia"]
TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DEMENTIA_CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
EIGHT_CLASS_NAMES = [f"tumor_{name}" for name in TUMOR_CLASSES] + [
    f"dementia_{name}" for name in DEMENTIA_CLASSES
]

TASK_CLASS_NAMES = {
    "binary": DOMAIN_CLASSES,
    "tumor": TUMOR_CLASSES,
    "dementia": DEMENTIA_CLASSES,
    "eight_class": EIGHT_CLASS_NAMES,
}

TASK_DEFAULT_ARCH = {
    "binary": "resnet50",
    "tumor": "efficientnet_b3",
    "dementia": "mobilenet_v3_large",
    "eight_class": "efficientnet_b3",
}

TASK_DEFAULT_MODEL_PATH = {
    "binary": MODELS_DIR / "binary_router.pt",
    "tumor": MODELS_DIR / "brain_tumor_classifier.pt",
    "dementia": MODELS_DIR / "alzheimers_classifier.pt",
    "eight_class": MODELS_DIR / "single_8class_classifier.pt",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    domain: str
    subtype: str
    source_split: str
    split: str = ""

    @property
    def eight_class(self) -> str:
        return f"{self.domain}_{self.subtype}"

    @property
    def binary_label(self) -> int:
        return DOMAIN_CLASSES.index(self.domain)


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(data_root: Path = DATA_ROOT) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    tumor_root = data_root / "brain_tumor"
    for source_name, source_split in (("Training", "pool"), ("Testing", "test")):
        for class_name in TUMOR_CLASSES:
            for path in iter_images(tumor_root / source_name / class_name):
                records.append(
                    ImageRecord(
                        path=path,
                        domain="tumor",
                        subtype=class_name,
                        source_split=source_split,
                    )
                )

    dementia_root = data_root / "alzheimers"
    for class_name in DEMENTIA_CLASSES:
        for path in iter_images(dementia_root / class_name):
            records.append(
                ImageRecord(
                    path=path,
                    domain="dementia",
                    subtype=class_name,
                    source_split="pool",
                )
            )

    return records


def split_counts(total: int, val_fraction: float, test_fraction: float, include_test: bool) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    test_count = int(round(total * test_fraction)) if include_test else 0
    val_count = int(round(total * val_fraction))

    if include_test and total >= 3:
        test_count = max(1, test_count)
    if total - test_count >= 2:
        val_count = max(1, val_count)

    if test_count + val_count >= total:
        overflow = test_count + val_count - total + 1
        val_count = max(0, val_count - overflow)

    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def duplicate_group_key(records: list[ImageRecord]) -> tuple[str, str, str]:
    """Return the splitting key for an exact-duplicate group.

    The key is only used to keep class balance roughly stratified while assigning
    whole duplicate groups to one split. If an exact image hash appears in the
    official tumor test folder, the full duplicate group is treated as test.
    """

    sorted_records = sorted(records, key=lambda record: str(record.path))
    if any(record.domain == "tumor" and record.source_split == "test" for record in sorted_records):
        tumor_test = [record for record in sorted_records if record.domain == "tumor" and record.source_split == "test"]
        anchor = tumor_test[0]
        return anchor.domain, anchor.subtype, "test"

    counter = Counter((record.domain, record.subtype, record.source_split) for record in sorted_records)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_split_units(records: Iterable[ImageRecord], *, dedupe_exact_hash: bool) -> list[list[ImageRecord]]:
    if not dedupe_exact_hash:
        return [[record] for record in records]

    buckets: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        buckets[sha256_file(record.path)].append(record)
    return list(buckets.values())


def assign_split(
    records: Iterable[ImageRecord],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    dedupe_exact_hash: bool = True,
) -> list[ImageRecord]:
    """Create strict splits before any augmentation is applied.

    Tumor images use the dataset's official Testing folder as the strict test
    set. Dementia images do not ship with an official split here, so they get a
    stratified deterministic train/val/test split. Exact duplicate image hashes
    are assigned as an indivisible group so the same pixels cannot cross splits.
    """

    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[list[ImageRecord]]] = defaultdict(list)
    assigned: list[ImageRecord] = []

    for unit in build_split_units(records, dedupe_exact_hash=dedupe_exact_hash):
        domain, subtype, source_split = duplicate_group_key(unit)
        grouped[(domain, subtype, source_split)].append(unit)

    for (domain, subtype, source_split), class_units in sorted(grouped.items()):
        shuffled = sorted(class_units, key=lambda unit: str(sorted(record.path for record in unit)[0]))
        rng.shuffle(shuffled)

        if domain == "tumor" and source_split == "test":
            for unit in shuffled:
                assigned.extend(record_with_split(record, "test") for record in unit)
            continue

        include_test = domain == "dementia"
        train_count, val_count, test_count = split_counts(
            len(shuffled),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            include_test=include_test,
        )

        train_records = shuffled[:train_count]
        val_records = shuffled[train_count : train_count + val_count]
        test_records = shuffled[train_count + val_count : train_count + val_count + test_count]

        for unit in train_records:
            assigned.extend(record_with_split(record, "train") for record in unit)
        for unit in val_records:
            assigned.extend(record_with_split(record, "val") for record in unit)
        for unit in test_records:
            assigned.extend(record_with_split(record, "test") for record in unit)

    return sorted(assigned, key=lambda item: (item.split, item.domain, item.subtype, str(item.path)))


def record_with_split(record: ImageRecord, split: str) -> ImageRecord:
    return ImageRecord(
        path=record.path,
        domain=record.domain,
        subtype=record.subtype,
        source_split=record.source_split,
        split=split,
    )


def create_manifest(
    output_path: Path = DEFAULT_MANIFEST,
    *,
    data_root: Path = DATA_ROOT,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    dedupe_exact_hash: bool = True,
) -> dict[str, object]:
    records = collect_records(data_root)
    assigned = assign_split(
        records,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        dedupe_exact_hash=dedupe_exact_hash,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "split",
                "domain",
                "subtype",
                "binary_label",
                "eight_class",
                "source_split",
            ],
        )
        writer.writeheader()
        for record in assigned:
            writer.writerow(
                {
                    "path": str(record.path.relative_to(PROJECT_ROOT)),
                    "split": record.split,
                    "domain": record.domain,
                    "subtype": record.subtype,
                    "binary_label": record.binary_label,
                    "eight_class": record.eight_class,
                    "source_split": record.source_split,
                }
            )

    summary = summarize_manifest(assigned)
    summary["manifest_path"] = str(output_path)
    summary["dedupe_exact_hash"] = dedupe_exact_hash
    return summary


def summarize_manifest(records: Iterable[Union[ImageRecord, dict[str, str]]]) -> dict[str, object]:
    split_counts_by_task: dict[str, Counter[str]] = {
        "all": Counter(),
        "binary": Counter(),
        "tumor": Counter(),
        "dementia": Counter(),
        "eight_class": Counter(),
    }
    class_counts: dict[str, Counter[str]] = {
        "binary": Counter(),
        "tumor": Counter(),
        "dementia": Counter(),
        "eight_class": Counter(),
    }

    for item in records:
        if isinstance(item, ImageRecord):
            split = item.split
            domain = item.domain
            subtype = item.subtype
            eight_class = item.eight_class
        else:
            split = item["split"]
            domain = item["domain"]
            subtype = item["subtype"]
            eight_class = item["eight_class"]

        split_counts_by_task["all"][split] += 1
        split_counts_by_task["binary"][split] += 1
        split_counts_by_task["eight_class"][split] += 1
        class_counts["binary"][f"{split}:{domain}"] += 1
        class_counts["eight_class"][f"{split}:{eight_class}"] += 1

        if domain == "tumor":
            split_counts_by_task["tumor"][split] += 1
            class_counts["tumor"][f"{split}:{subtype}"] += 1
        elif domain == "dementia":
            split_counts_by_task["dementia"][split] += 1
            class_counts["dementia"][f"{split}:{subtype}"] += 1

    return {
        "splits": {key: dict(value) for key, value in split_counts_by_task.items()},
        "class_counts": {key: dict(value) for key, value in class_counts.items()},
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def label_for_row(row: dict[str, str], task: str) -> Optional[int]:
    if task == "binary":
        return DOMAIN_CLASSES.index(row["domain"])
    if task == "tumor":
        return TUMOR_CLASSES.index(row["subtype"]) if row["domain"] == "tumor" else None
    if task == "dementia":
        return DEMENTIA_CLASSES.index(row["subtype"]) if row["domain"] == "dementia" else None
    if task == "eight_class":
        return EIGHT_CLASS_NAMES.index(row["eight_class"])
    raise ValueError(f"Unsupported task: {task}")


def select_samples(
    manifest_path: Path,
    *,
    split: str,
    task: str,
    max_samples_per_class: Optional[int] = None,
) -> list[tuple[Path, int]]:
    rows = [row for row in read_manifest(manifest_path) if row["split"] == split]
    buckets: dict[int, list[tuple[Path, int]]] = defaultdict(list)

    for row in rows:
        label = label_for_row(row, task)
        if label is None:
            continue
        buckets[label].append((PROJECT_ROOT / row["path"], label))

    selected: list[tuple[Path, int]] = []
    for label in sorted(buckets):
        samples = sorted(buckets[label], key=lambda sample: str(sample[0]))
        if max_samples_per_class is not None:
            samples = samples[:max_samples_per_class]
        selected.extend(samples)

    return selected


def build_transforms(train: bool, image_size: int, random_erasing: float = 0.0):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        steps = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.12, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
        if random_erasing > 0:
            steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
        return transforms.Compose(steps)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ManifestImageDataset:
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image), torch.tensor(label, dtype=torch.long)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {path}") from exc


def build_model(arch: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features, num_classes))
        return model

    if arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
        return model

    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


def class_weights(samples: list[tuple[Path, int]], num_classes: int):
    import torch

    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    weights = [
        total / (num_classes * counts[class_idx]) if counts[class_idx] else 0.0
        for class_idx in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_loader(model, loader, criterion, device) -> dict[str, object]:
    import torch

    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / max(1, len(y_true))
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_metrics(metrics: dict[str, object], class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    except Exception:
        report = {"accuracy": metrics["accuracy"]}
        cm = [[0 for _ in class_names] for _ in class_names]
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1

    payload = {
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "n": len(y_true),
        "correct": sum(1 for true, pred in zip(y_true, y_pred) if true == pred),
        "classification_report": report,
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "class_names": class_names,
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "true_index", "pred_index", "true_label", "pred_label"])
        for index, (true, pred) in enumerate(zip(y_true, y_pred)):
            writer.writerow([index, true, pred, class_names[true], class_names[pred]])

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *list(row)])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.75)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
        plt.close()
    except Exception:
        pass


def train_task(args: argparse.Namespace) -> Path:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        create_manifest(output_path=manifest_path)

    task = args.task
    class_names = TASK_CLASS_NAMES[task]
    image_size = args.image_size
    max_per_class = args.max_samples_per_class
    if args.smoke_test and max_per_class is None:
        max_per_class = 16

    train_samples = select_samples(
        manifest_path,
        split="train",
        task=task,
        max_samples_per_class=max_per_class,
    )
    val_samples = select_samples(
        manifest_path,
        split="val",
        task=task,
        max_samples_per_class=max_per_class,
    )

    if not train_samples or not val_samples:
        raise SystemExit(f"No train/val samples found for task '{task}'. Check {manifest_path}.")

    train_dataset = ManifestImageDataset(
        train_samples,
        build_transforms(train=True, image_size=image_size, random_erasing=args.random_erasing),
    )
    val_dataset = ManifestImageDataset(val_samples, build_transforms(train=False, image_size=image_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = args.arch or TASK_DEFAULT_ARCH[task]
    model = build_model(arch, len(class_names), pretrained=args.pretrained).to(device)

    weights = class_weights(train_samples, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_dir = Path(args.output_dir) / task / dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_patience_score = -1.0
    epochs_without_improvement = 0
    best_checkpoint = Path(args.model_out or TASK_DEFAULT_MODEL_PATH[task])
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"{task} epoch {epoch + 1}/{args.epochs}", leave=False)
        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_dataset))
        train_acc = correct / max(1, total)
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        val_acc = float(val_metrics["accuracy"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": val_acc,
            }
        )

        print(
            f"{task} epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            torch.save(
                {
                    "task": task,
                    "arch": arch,
                    "class_names": class_names,
                    "image_size": image_size,
                    "model_state": model.state_dict(),
                    "history": history,
                    "training_config": {
                        "epochs_requested": args.epochs,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "label_smoothing": args.label_smoothing,
                        "random_erasing": args.random_erasing,
                        "early_stop_patience": args.early_stop_patience,
                        "early_stop_min_delta": args.early_stop_min_delta,
                    },
                    "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                best_checkpoint,
            )
            print(f"Saved best checkpoint to {best_checkpoint}", flush=True)

        if val_acc > best_patience_score + args.early_stop_min_delta:
            best_patience_score = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping {task} after {epoch + 1} epochs; "
                f"no validation improvement for {epochs_without_improvement} epochs.",
                flush=True,
            )
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    test_task(args, checkpoint_path=best_checkpoint, output_dir=run_dir / "test")
    return best_checkpoint


def load_checkpoint_model(checkpoint_path: Path, device):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a pipeline checkpoint. "
            "Retrain with src/experiment_pipeline.py or convert the model first."
        )

    class_names = checkpoint["class_names"]
    model = build_model(
        checkpoint.get("arch", "resnet50"),
        len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    return model, checkpoint


def test_task(
    args: argparse.Namespace,
    *,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, object]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    checkpoint_path = checkpoint_path or Path(args.checkpoint or args.model_out or TASK_DEFAULT_MODEL_PATH[args.task])
    manifest_path = Path(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", args.image_size))

    split = getattr(args, "split", "test")
    eval_limit = None
    if args.smoke_test:
        eval_limit = args.max_samples_per_class or 16

    samples = select_samples(
        manifest_path,
        split=split,
        task=args.task,
        max_samples_per_class=eval_limit,
    )
    dataset = ManifestImageDataset(samples, build_transforms(train=False, image_size=image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_loader(model, loader, criterion, device)

    output_dir = output_dir or Path(args.output_dir) / args.task / f"{split}_evaluation"
    save_metrics(metrics, class_names, output_dir)
    print(f"Saved {args.task} {split} metrics to {output_dir}", flush=True)
    return metrics


def evaluate_hierarchical(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary_model, binary_ckpt = load_checkpoint_model(Path(args.binary_checkpoint), device)
    tumor_model, tumor_ckpt = load_checkpoint_model(Path(args.tumor_checkpoint), device)
    dementia_model, dementia_ckpt = load_checkpoint_model(Path(args.dementia_checkpoint), device)

    binary_tf = build_transforms(train=False, image_size=int(binary_ckpt.get("image_size", args.image_size)))
    tumor_tf = build_transforms(train=False, image_size=int(tumor_ckpt.get("image_size", args.image_size)))
    dementia_tf = build_transforms(train=False, image_size=int(dementia_ckpt.get("image_size", args.image_size)))

    rows = [row for row in read_manifest(Path(args.manifest)) if row["split"] == args.split]
    if args.smoke_test:
        limit = args.max_samples_per_class or 16
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            buckets[row["eight_class"]].append(row)
        rows = []
        for class_name in sorted(buckets):
            rows.extend(sorted(buckets[class_name], key=lambda item: item["path"])[:limit])

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for row in rows:
            image = Image.open(PROJECT_ROOT / row["path"]).convert("RGB")
            binary_logits = binary_model(binary_tf(image).unsqueeze(0).to(device))
            domain_idx = int(torch.argmax(binary_logits, dim=1).item())
            predicted_domain = binary_ckpt["class_names"][domain_idx]

            if predicted_domain == "tumor":
                logits = tumor_model(tumor_tf(image).unsqueeze(0).to(device))
                subtype = tumor_ckpt["class_names"][int(torch.argmax(logits, dim=1).item())]
                predicted_label = f"tumor_{subtype}"
            else:
                logits = dementia_model(dementia_tf(image).unsqueeze(0).to(device))
                subtype = dementia_ckpt["class_names"][int(torch.argmax(logits, dim=1).item())]
                predicted_label = f"dementia_{subtype}"

            y_true.append(EIGHT_CLASS_NAMES.index(row["eight_class"]))
            y_pred.append(EIGHT_CLASS_NAMES.index(predicted_label))

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / max(1, len(y_true))
    output_dir = Path(args.output_dir) / "hierarchical" / f"{args.split}_evaluation"
    save_metrics(
        {
            "accuracy": accuracy,
            "loss": 0.0,
            "y_true": y_true,
            "y_pred": y_pred,
        },
        EIGHT_CLASS_NAMES,
        output_dir,
    )
    print(f"Saved hierarchical {args.split} metrics to {output_dir}", flush=True)


def command_manifest(args: argparse.Namespace) -> None:
    summary = create_manifest(
        output_path=Path(args.manifest),
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        dedupe_exact_hash=args.dedupe_exact_hash,
    )
    print(json.dumps(summary, indent=2))


def command_summary(args: argparse.Namespace) -> None:
    rows = read_manifest(Path(args.manifest))
    print(json.dumps(summarize_manifest(rows), indent=2))


def command_suite(args: argparse.Namespace) -> None:
    if not Path(args.manifest).exists():
        create_manifest(output_path=Path(args.manifest))

    original_task = args.task
    for task in ["binary", "tumor", "dementia"]:
        args.task = task
        args.arch = None
        args.model_out = str(TASK_DEFAULT_MODEL_PATH[task])
        train_task(args)

    args.split = "test"
    args.binary_checkpoint = str(TASK_DEFAULT_MODEL_PATH["binary"])
    args.tumor_checkpoint = str(TASK_DEFAULT_MODEL_PATH["tumor"])
    args.dementia_checkpoint = str(TASK_DEFAULT_MODEL_PATH["dementia"])
    evaluate_hierarchical(args)

    args.task = "eight_class"
    args.arch = None
    args.model_out = str(TASK_DEFAULT_MODEL_PATH["eight_class"])
    train_task(args)
    args.task = original_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
        subparser.add_argument("--output-dir", default=str(EXPERIMENTS_DIR))
        subparser.add_argument("--image-size", type=int, default=224)
        subparser.add_argument("--batch-size", type=int, default=32)
        subparser.add_argument("--num-workers", type=int, default=0)
        subparser.add_argument("--max-samples-per-class", type=int)
        subparser.add_argument("--smoke-test", action="store_true")

    manifest = subparsers.add_parser("create-manifest", help="Create strict train/val/test manifest.")
    manifest.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--val-fraction", type=float, default=0.1)
    manifest.add_argument("--test-fraction", type=float, default=0.2)
    manifest.add_argument(
        "--allow-duplicate-leakage",
        dest="dedupe_exact_hash",
        action="store_false",
        default=True,
        help="Legacy mode: do not group exact duplicate image hashes into one split.",
    )
    manifest.set_defaults(func=command_manifest)

    summary = subparsers.add_parser("summary", help="Summarize an existing manifest.")
    summary.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    summary.set_defaults(func=command_summary)

    train = subparsers.add_parser("train", help="Train one task and evaluate on strict test split.")
    add_common(train)
    train.add_argument("--task", choices=sorted(TASK_CLASS_NAMES), required=True)
    train.add_argument("--arch", choices=["resnet50", "efficientnet_b3", "mobilenet_v3_large"])
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument("--random-erasing", type=float, default=0.0)
    train.add_argument("--early-stop-patience", type=int, default=0)
    train.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train.add_argument("--model-out")
    train.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    train.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    train.set_defaults(func=train_task)

    test = subparsers.add_parser("test", help="Evaluate one checkpoint on a manifest split.")
    add_common(test)
    test.add_argument("--task", choices=sorted(TASK_CLASS_NAMES), required=True)
    test.add_argument("--checkpoint")
    test.add_argument("--model-out")
    test.add_argument("--split", choices=["train", "val", "test"], default="test")
    test.set_defaults(func=test_task)

    hierarchical = subparsers.add_parser("evaluate-hierarchical", help="Evaluate binary router plus specialists.")
    add_common(hierarchical)
    hierarchical.add_argument("--split", choices=["train", "val", "test"], default="test")
    hierarchical.add_argument("--binary-checkpoint", default=str(TASK_DEFAULT_MODEL_PATH["binary"]))
    hierarchical.add_argument("--tumor-checkpoint", default=str(TASK_DEFAULT_MODEL_PATH["tumor"]))
    hierarchical.add_argument("--dementia-checkpoint", default=str(TASK_DEFAULT_MODEL_PATH["dementia"]))
    hierarchical.set_defaults(func=evaluate_hierarchical)

    suite = subparsers.add_parser("suite", help="Run binary, specialist, and 8-class training.")
    add_common(suite)
    suite.add_argument("--task", default="binary")
    suite.add_argument("--epochs", type=int, default=20)
    suite.add_argument("--learning-rate", type=float, default=1e-4)
    suite.add_argument("--weight-decay", type=float, default=1e-4)
    suite.add_argument("--label-smoothing", type=float, default=0.0)
    suite.add_argument("--random-erasing", type=float, default=0.0)
    suite.add_argument("--early-stop-patience", type=int, default=0)
    suite.add_argument("--early-stop-min-delta", type=float, default=0.0)
    suite.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    suite.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    suite.set_defaults(func=command_suite)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
