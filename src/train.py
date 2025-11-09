import argparse
import json
import math
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4, val_split: float = 0.1):
    # ImageNet normalization constants (match EfficientNet pretraining)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(size=300, scale=(0.75, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(320, antialias=True),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_root = os.path.join(data_dir, "Training")
    test_root = os.path.join(data_dir, "Testing")
    if not os.path.isdir(train_root) or not os.path.isdir(test_root):
        raise FileNotFoundError(
            f"Could not find expected 'Training' and 'Testing' folders in {data_dir}"
        )

    full_train_train = datasets.ImageFolder(train_root, transform=train_tfms)
    full_train_eval = datasets.ImageFolder(train_root, transform=eval_tfms)

    # Split train into train/val (by indices)
    n_total = len(full_train_train)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(range(n_total), [n_train, n_val])
    train_set = torch.utils.data.Subset(full_train_train, train_subset.indices)
    val_set = torch.utils.data.Subset(full_train_eval, val_subset.indices)

    test_set = datasets.ImageFolder(test_root, transform=eval_tfms)

    # Class names
    class_names = full_train_train.classes

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes: int):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    # Replace head
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
        loss_sum += loss.item() * targets.size(0)
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return avg_loss, acc


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    # Try DirectML (Windows GPU via DX12)
    try:
        import torch_directml as dml  # type: ignore
        return dml.device(), "dml"
    except Exception:
        return torch.device("cpu"), "cpu"


def train(
    data_dir: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    label_smoothing: float = 0.1,
):
    seed_everything(42)

    device, device_kind = get_device()
    print(f"Using device: {device_kind}")

    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir, batch_size, num_workers=num_workers
    )

    model = build_model(num_classes=len(class_names)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_kind == "cuda"))

    best_val_acc = 0.0
    best_path = os.path.join(out_dir, "brain_tumor_classifier.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device_kind == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_loss += loss.item() * targets.size(0)
            total += targets.numel()

        train_loss = running_loss / max(1, total)
        train_acc = running_correct / max(1, total)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": len(class_names),
                "class_names": class_names,
                "arch": "efficientnet_b3",
                "timestamp": datetime.now().isoformat(),
            }, best_path)
            print(f"Saved new best model to {best_path} (val acc={best_val_acc:.4f})")

    # Final test evaluation
    print("Evaluating best model on test set...")
    checkpoint = torch.load(best_path, map_location=device)
    model = build_model(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"]) 
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save metadata
    meta = {
        "class_names": checkpoint["class_names"],
        "arch": checkpoint["arch"],
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    with open(os.path.join(out_dir, "brain_tumor_classifier_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Metadata written.")


def parse_args():
    p = argparse.ArgumentParser(description="Train brain tumor MRI classifier (GPU-accelerated)")
    p.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("..", "data", "brain-tumor-mri-dataset"),
        help="Folder containing Training/ and Testing/",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("..", "models"),
        help="Output directory for model checkpoints and metadata",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        label_smoothing=args.label_smoothing,
    )
