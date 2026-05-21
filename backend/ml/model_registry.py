"""Model Registry - manages loading and inference for all detection models.

Static configs declare the legacy ResNet50 heads. Trained MedMNIST models
(those with weights + metrics in backend/ml/weights/) are discovered at
runtime and merged in.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Static configurations for the legacy named slots
MODEL_CONFIGS = {
    "resnet50_breast": {
        "display_name": "ResNet-50 (Breast Cancer)",
        "classes": ["Normal", "Malignant"],
        "image_type": "breast",
        "input_size": 224,
        "architecture": "resnet50",
        "description": "ResNet-50 fine-tuned for breast histopathology classification",
    },
    "resnet50_lung": {
        "display_name": "ResNet-50 (Lung Cancer)",
        "classes": ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"],
        "image_type": "lung",
        "input_size": 224,
        "architecture": "resnet50",
        "description": "ResNet-50 for lung cancer subtype classification",
    },
    "resnet50_brain": {
        "display_name": "ResNet-50 (Brain Tumor)",
        "classes": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
        "image_type": "brain",
        "input_size": 224,
        "architecture": "resnet50",
        "description": "ResNet-50 for brain tumor MRI classification",
    },
    "resnet50_skin": {
        "display_name": "ResNet-50 (Skin Lesion)",
        "classes": ["Benign", "Malignant"],
        "image_type": "skin",
        "input_size": 224,
        "architecture": "resnet50",
        "description": "ResNet-50 for dermoscopic skin lesion classification",
    },
}


def _discover_trained_models() -> dict:
    """Scan WEIGHTS_DIR for {model}.metrics.json + {model}.pth pairs."""
    if not WEIGHTS_DIR.exists():
        return {}
    found = {}
    for meta in WEIGHTS_DIR.glob("*.metrics.json"):
        try:
            with open(meta) as f:
                m = json.load(f)
            model_name = m.get("model_name") or meta.stem.replace(".metrics", "")
            weights = WEIGHTS_DIR / f"{model_name}.pth"
            if not weights.exists():
                continue
            arch = m.get("architecture", "resnet18")
            found[model_name] = {
                "display_name": f"{arch.title()} ({m.get('dataset', 'MedMNIST').replace('mnist', '').title()})",
                "classes": m.get("labels", []),
                "image_type": m.get("dataset", "medmnist"),
                "input_size": 64 if arch != "resnet50" else 224,
                "architecture": arch,
                "description": f"Trained on {m.get('dataset', 'MedMNIST')} — {m.get('accuracy', 0):.1%} test accuracy",
                "trained": True,
                "accuracy": m.get("accuracy"),
                "dataset": m.get("dataset"),
            }
        except Exception:
            continue
    return found


def get_all_configs() -> dict:
    """Static configs + dynamically discovered trained models."""
    return {**MODEL_CONFIGS, **_discover_trained_models()}


# In-memory model cache
_loaded_models: dict = {}


def get_available_models() -> list[dict]:
    """Return list of available model configurations (legacy + trained)."""
    return [
        {"model_name": name, **config}
        for name, config in get_all_configs().items()
    ]


def get_model_config(model_name: str) -> Optional[dict]:
    return get_all_configs().get(model_name)


def _build_model(architecture: str, num_classes: int):
    """Construct an untrained model of the requested architecture."""
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
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )


def _load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    config = get_model_config(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        import torch

        num_classes = len(config["classes"])
        architecture = config.get("architecture", "resnet50")
        model = _build_model(architecture, num_classes)
        model.eval()

        # Try {WEIGHTS_DIR}/{model_name}.pth first, then settings.MODEL_DIR/{model_name}.pth
        candidates = [WEIGHTS_DIR / f"{model_name}.pth"]
        try:
            from config.settings import settings
            candidates.append(settings.MODEL_DIR / f"{model_name}.pth")
        except Exception:
            pass

        loaded = False
        for path in candidates:
            if path.exists():
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                try:
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded fine-tuned weights for {model_name} from {path}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Weights at {path} do not match model architecture: {e}")
                    continue

        if not loaded:
            logger.info(f"No fine-tuned weights for {model_name}; using ImageNet features only")

        _loaded_models[model_name] = model
        return model

    except ImportError:
        logger.warning("PyTorch not available, using mock predictions")
        _loaded_models[model_name] = None
        return None


def _preprocess_image(image_path: str, input_size: int = 224) -> "torch.Tensor":
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def reset_cache() -> None:
    """Drop the in-memory model cache. Used after training to pick up fresh weights."""
    _loaded_models.clear()


def predict(model_name: str, image_path: str) -> dict:
    """Run inference. Returns dict with prediction_label, confidence, all_probabilities, processing_time_ms."""
    config = get_model_config(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    start = time.time()
    model = _load_model(model_name)

    if model is None:
        import random
        classes = config["classes"]
        idx = random.randint(0, len(classes) - 1)
        probs = [random.random() for _ in classes]
        total = sum(probs)
        probs = [p / total for p in probs]
        probs[idx] = max(probs)
        return {
            "prediction_label": classes[idx],
            "confidence": round(probs[idx], 4),
            "all_probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    import torch
    import torch.nn.functional as F

    tensor = _preprocess_image(image_path, config.get("input_size", 224))
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    # Handle scalar case (single-output binary)
    if probs.ndim == 0:
        probs = np.array([1 - float(probs), float(probs)])

    classes = config["classes"]
    idx = int(np.argmax(probs))

    return {
        "prediction_label": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "processing_time_ms": int((time.time() - start) * 1000),
    }
