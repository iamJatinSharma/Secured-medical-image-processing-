"""Real PyTorch trainer for MedMNIST datasets.

Trains a ResNet18 (default) or small CNN on a MedMNIST dataset, streams
per-epoch metrics via a callback, saves weights + metrics to
backend/ml/weights/{model_name}.{pth,metrics.json}.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Map MedMNIST dataset name -> human-readable info shown in UI
DATASET_INFO = {
    "breastmnist": {
        "name": "BreastMNIST",
        "modality": "Mammography (Breast Ultrasound)",
        "task": "Binary classification",
        "classes": ["malignant", "normal/benign"],
        "size_train": 546,
        "size_val": 78,
        "size_test": 156,
    },
    "pneumoniamnist": {
        "name": "PneumoniaMNIST",
        "modality": "Chest X-ray",
        "task": "Binary classification",
        "classes": ["normal", "pneumonia"],
        "size_train": 4708,
        "size_val": 524,
        "size_test": 624,
    },
    "dermamnist": {
        "name": "DermaMNIST",
        "modality": "Dermatoscope",
        "task": "Multi-class (7)",
        "classes": ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"],
        "size_train": 7007,
        "size_val": 1003,
        "size_test": 2005,
    },
    "bloodmnist": {
        "name": "BloodMNIST",
        "modality": "Blood cell microscope",
        "task": "Multi-class (8)",
        "classes": ["basophil", "eosinophil", "erythroblast", "immature granulocytes", "lymphocyte", "monocyte", "neutrophil", "platelet"],
        "size_train": 11959,
        "size_val": 1712,
        "size_test": 3421,
    },
    "octmnist": {
        "name": "OCTMNIST",
        "modality": "Retinal OCT",
        "task": "Multi-class (4)",
        "classes": ["choroidal neovascularization", "diabetic macular edema", "drusen", "normal"],
        "size_train": 97477,
        "size_val": 10832,
        "size_test": 1000,
    },
    "pathmnist": {
        "name": "PathMNIST",
        "modality": "Colon pathology",
        "task": "Multi-class (9)",
        "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        "size_train": 89996,
        "size_val": 10004,
        "size_test": 7180,
    },
}

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


@dataclass
class EpochMetrics:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    percent_complete: float
    eta_seconds: float


@dataclass
class TrainArgs:
    dataset: str = "breastmnist"
    architecture: str = "resnet18"  # resnet18 | resnet50 | smallcnn
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    image_size: int = 64  # MedMNIST native is 28; resize for ResNet
    max_train_samples: Optional[int] = None
    model_name: str = ""  # output weights filename — derived if empty
    history: list = field(default_factory=list)


def list_datasets() -> list[dict]:
    """Return UI-friendly dataset list."""
    return [{"id": k, **v} for k, v in DATASET_INFO.items()]


def _build_model(architecture: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if architecture == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(m.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        return m
    if architecture == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        return m
    # smallcnn — fast CPU fallback
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )


def _build_dataloaders(dataset_name: str, image_size: int, batch_size: int, max_train_samples: Optional[int]):
    import medmnist
    from medmnist import INFO
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    if dataset_name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {dataset_name}")

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    # MedMNIST gives uint8 [0,255] RGB or grayscale. Convert grayscale -> 3ch.
    n_channels = info["n_channels"]

    base_tx = [transforms.ToTensor()]
    if n_channels == 1:
        base_tx.append(transforms.Lambda(lambda t: t.repeat(3, 1, 1)))
    base_tx.append(transforms.Resize((image_size, image_size), antialias=True))
    base_tx.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(base_tx)

    train_ds = DataClass(split="train", transform=transform, download=True)
    val_ds = DataClass(split="val", transform=transform, download=True)
    test_ds = DataClass(split="test", transform=transform, download=True)

    if max_train_samples and max_train_samples < len(train_ds):
        idx = np.random.RandomState(42).choice(len(train_ds), size=max_train_samples, replace=False).tolist()
        train_ds = Subset(train_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(info["label"])
    class_names = [info["label"][str(i)] for i in range(num_classes)]
    return train_loader, val_loader, test_loader, num_classes, class_names


def _epoch_loop(model, loader, criterion, optimizer, device, training: bool):
    import torch

    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.long().squeeze(-1).to(device)  # MedMNIST labels are [N,1]

            if training:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def _evaluate_full(model, loader, device, num_classes: int):
    """Run inference on a loader, return (y_true, y_pred, y_probs)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().squeeze(-1).to(device)
            out = model(x)
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    return all_true, all_pred, np.concatenate(all_probs, axis=0)


def _compute_test_metrics(y_true, y_pred, y_probs, class_names):
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
        roc_curve, auc,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    cr = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    # Normalize precision/recall/f1-score keys for frontend consumption
    clean_report = {}
    for label, vals in cr.items():
        if isinstance(vals, dict):
            clean_report[label] = {
                "precision": float(vals.get("precision", 0)),
                "recall": float(vals.get("recall", 0)),
                "f1-score": float(vals.get("f1-score", 0)),
                "support": int(vals.get("support", 0)),
            }

    result = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "labels": class_names,
        "classification_report": clean_report,
    }

    # ROC + AUC (binary case only — averaged for multi-class would need explicit handling)
    if len(class_names) == 2 and y_probs is not None:
        positive_probs = y_probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true, positive_probs)
        roc_auc = float(auc(fpr, tpr))
        # Sub-sample for compact JSON
        if len(fpr) > 50:
            idx = np.linspace(0, len(fpr) - 1, 50).astype(int)
            fpr = fpr[idx]
            tpr = tpr[idx]
        result["auc"] = roc_auc
        result["roc_data"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    return result


def train(
    args: TrainArgs,
    on_epoch: Optional[Callable[[EpochMetrics], None]] = None,
) -> dict:
    """Train a model and save weights + metrics.

    Returns a dict with: model_name, accuracy, f1_score, weights_path, metrics_path.
    The on_epoch callback is invoked with EpochMetrics after each epoch (used by
    the WebSocket broadcaster).
    """
    import torch
    import torch.nn as nn

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.model_name:
        args.model_name = f"{args.architecture}_{args.dataset}"

    logger.info(f"Training {args.model_name} on {args.dataset} for {args.epochs} epochs")

    train_loader, val_loader, test_loader, num_classes, class_names = _build_dataloaders(
        args.dataset, args.image_size, args.batch_size, args.max_train_samples,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.architecture, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    epochs_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _epoch_loop(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = _epoch_loop(model, val_loader, criterion, optimizer, device, training=False)

        elapsed = time.time() - start_time
        per_epoch = elapsed / epoch
        eta = per_epoch * (args.epochs - epoch)
        percent = (epoch / args.epochs) * 100

        epochs_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        })

        if on_epoch:
            on_epoch(EpochMetrics(
                epoch=epoch,
                total_epochs=args.epochs,
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
                train_acc=round(train_acc, 4),
                val_acc=round(val_acc, 4),
                percent_complete=round(percent, 1),
                eta_seconds=round(eta, 1),
            ))

    total_time = round(time.time() - start_time, 2)
    logger.info(f"Training complete in {total_time}s")

    # Test-set evaluation
    y_true, y_pred, y_probs = _evaluate_full(model, test_loader, device, num_classes)
    metrics = _compute_test_metrics(y_true, y_pred, y_probs, class_names)
    metrics["model_name"] = args.model_name
    metrics["dataset"] = args.dataset
    metrics["architecture"] = args.architecture
    metrics["epochs"] = args.epochs
    metrics["total_time_seconds"] = total_time
    metrics["training_history"] = {
        "epochs": [h["epoch"] for h in epochs_history],
        "train_loss": [h["train_loss"] for h in epochs_history],
        "val_loss": [h["val_loss"] for h in epochs_history],
        "train_acc": [h["train_acc"] for h in epochs_history],
        "val_acc": [h["val_acc"] for h in epochs_history],
    }

    weights_path = WEIGHTS_DIR / f"{args.model_name}.pth"
    metrics_path = WEIGHTS_DIR / f"{args.model_name}.metrics.json"
    torch.save(model.state_dict(), weights_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_name": args.model_name,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "best_val_acc": max(h["val_acc"] for h in epochs_history),
        "total_time_seconds": total_time,
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "labels": class_names,
        "architecture": args.architecture,
    }


def list_trained_models() -> list[dict]:
    """Discover models that have saved weights + metrics."""
    if not WEIGHTS_DIR.exists():
        return []
    out = []
    for meta in WEIGHTS_DIR.glob("*.metrics.json"):
        try:
            with open(meta) as f:
                m = json.load(f)
            weights = WEIGHTS_DIR / f"{m.get('model_name', meta.stem.replace('.metrics', ''))}.pth"
            if weights.exists():
                out.append({
                    "model_name": m.get("model_name"),
                    "dataset": m.get("dataset"),
                    "architecture": m.get("architecture"),
                    "accuracy": m.get("accuracy"),
                    "labels": m.get("labels", []),
                })
        except Exception:
            continue
    return out


def load_metrics(model_name: str) -> Optional[dict]:
    """Load saved test metrics for a trained model."""
    metrics_path = WEIGHTS_DIR / f"{model_name}.metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None
