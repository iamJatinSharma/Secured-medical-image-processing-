"""Evaluation metrics API endpoints."""
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from typing import Optional
import numpy as np
from PIL import Image
import io
import json

from api.deps import get_current_user
from db.models import User

router = APIRouter()


def _read_upload_image(file_bytes: bytes) -> np.ndarray:
    """Read uploaded file bytes into numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


@router.get("/health")
async def health():
    return {"status": "ok", "module": "metrics"}


@router.post("/compare")
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Compare two images — returns PSNR, SSIM, MSE."""
    from ml.metrics import calculate_psnr, calculate_ssim, calculate_mse

    bytes1 = await image1.read()
    bytes2 = await image2.read()
    img1 = _read_upload_image(bytes1)
    img2 = _read_upload_image(bytes2)

    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((w, h)))
        img2 = np.array(Image.fromarray(img2).resize((w, h)))

    return {
        "psnr_db": calculate_psnr(img1, img2),
        "ssim": calculate_ssim(img1, img2),
        "mse": calculate_mse(img1, img2),
        "identical": calculate_mse(img1, img2) == 0,
    }


@router.post("/model-performance")
async def model_performance(
    y_true: str = Form(...),
    y_pred: str = Form(...),
    y_probs: Optional[str] = Form(None),
    labels: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """Calculate model performance metrics from true/predicted labels."""
    from ml.metrics import (
        calculate_f1, calculate_auc,
        generate_confusion_matrix, generate_classification_report, generate_roc_data,
    )

    try:
        true_labels = json.loads(y_true)
        pred_labels = json.loads(y_pred)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for y_true or y_pred")

    label_list = json.loads(labels) if labels else None
    probs = json.loads(y_probs) if y_probs else None

    result = {
        "accuracy": sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels) if true_labels else 0,
        "f1_score": calculate_f1(true_labels, pred_labels),
        "confusion_matrix": generate_confusion_matrix(true_labels, pred_labels, label_list),
        "classification_report": generate_classification_report(true_labels, pred_labels, label_list),
    }

    if probs:
        result["auc"] = calculate_auc(true_labels, probs)
        result["roc_data"] = generate_roc_data(true_labels, probs)

    return result


@router.get("/model/{model_name}")
async def get_model_metrics(
    model_name: str,
    current_user: User = Depends(get_current_user),
):
    """Get cached performance metrics for a model.

    Returns real metrics from disk if the model has been trained,
    otherwise falls back to demo metrics for legacy ResNet50 slots.
    """
    # Prefer real metrics saved by trainer
    from ml.trainer import load_metrics
    saved = load_metrics(model_name)
    if saved:
        return saved

    demo_metrics = {
        "resnet50_breast": {
            "model_name": "resnet50_breast",
            "accuracy": 0.923,
            "f1_score": 0.918,
            "auc": 0.961,
            "confusion_matrix": [[142, 13], [8, 137]],
            "labels": ["Normal", "Cancer"],
            "classification_report": {
                "Normal": {"precision": 0.947, "recall": 0.916, "f1-score": 0.931, "support": 155},
                "Cancer": {"precision": 0.913, "recall": 0.945, "f1-score": 0.929, "support": 145},
            },
            "roc_data": {
                "fpr": [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.30, 0.50, 1.0],
                "tpr": [0.0, 0.45, 0.72, 0.83, 0.90, 0.94, 0.97, 0.99, 1.0],
                "auc": 0.961,
            },
            "training_history": {
                "epochs": list(range(1, 31)),
                "train_loss": [0.693, 0.584, 0.478, 0.401, 0.342, 0.298, 0.261, 0.231, 0.207, 0.188,
                               0.172, 0.159, 0.148, 0.139, 0.131, 0.124, 0.118, 0.113, 0.109, 0.105,
                               0.102, 0.099, 0.096, 0.094, 0.092, 0.090, 0.088, 0.087, 0.086, 0.085],
                "val_loss": [0.720, 0.612, 0.510, 0.438, 0.385, 0.348, 0.318, 0.295, 0.278, 0.265,
                             0.255, 0.248, 0.243, 0.239, 0.236, 0.234, 0.233, 0.232, 0.232, 0.232,
                             0.233, 0.233, 0.234, 0.235, 0.235, 0.236, 0.237, 0.238, 0.239, 0.239],
                "train_acc": [0.52, 0.61, 0.70, 0.76, 0.81, 0.84, 0.87, 0.89, 0.90, 0.91,
                              0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95,
                              0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97],
                "val_acc": [0.48, 0.57, 0.65, 0.72, 0.77, 0.81, 0.84, 0.86, 0.87, 0.88,
                            0.89, 0.90, 0.90, 0.91, 0.91, 0.91, 0.92, 0.92, 0.92, 0.92,
                            0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92],
            },
        },
        "resnet50_lung": {
            "model_name": "resnet50_lung",
            "accuracy": 0.891,
            "f1_score": 0.885,
            "auc": 0.943,
            "confusion_matrix": [[135, 18], [11, 136]],
            "labels": ["Normal", "Pneumonia"],
        },
        "resnet50_brain": {
            "model_name": "resnet50_brain",
            "accuracy": 0.905,
            "f1_score": 0.901,
            "auc": 0.952,
            "confusion_matrix": [[140, 14], [10, 136]],
            "labels": ["Normal", "Tumor"],
        },
        "resnet50_skin": {
            "model_name": "resnet50_skin",
            "accuracy": 0.878,
            "f1_score": 0.872,
            "auc": 0.935,
            "confusion_matrix": [[130, 20], [15, 135]],
            "labels": ["Benign", "Malignant"],
        },
    }

    if model_name in demo_metrics:
        return demo_metrics[model_name]

    raise HTTPException(status_code=404, detail=f"No metrics found for model: {model_name}")
