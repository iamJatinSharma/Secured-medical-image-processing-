"""Admin endpoints — seed demo data for the current user.

Not real "admin" auth: protected by a SEED_SECRET header so anyone with
the secret + a valid JWT can populate their own dashboard with sample
images, predictions, and reports. Useful before a demo when the
ephemeral free-tier SQLite has just been wiped.
"""
from __future__ import annotations

import os
import random
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_user
from config.settings import settings
from db.crud import create_image, create_prediction, create_report
from db.database import get_db
from db.models import Image, ImageType, Prediction, Report, User

router = APIRouter()

BREAST_LABELS = ["Normal", "Malignant"]
MODEL_NAME = "resnet50_breast"
NUM_IMAGES = 8
NUM_REPORTS = 2


def _make_synthetic_image(path: Path, label: str) -> tuple[int, int, int, int]:
    """Write a 224x224 RGB PNG tinted by class. Returns (w, h, channels, bytes)."""
    from PIL import Image as PILImage
    import numpy as np

    rng = np.random.default_rng()
    # Class-tinted base + noise so the images visually differ
    if label == "Malignant":
        base = np.array([180, 60, 80], dtype=np.uint8)  # reddish
    else:
        base = np.array([90, 140, 200], dtype=np.uint8)  # bluish
    noise = rng.integers(-30, 30, size=(224, 224, 3), dtype=np.int16)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = PILImage.fromarray(arr, mode="RGB")
    img.save(path, format="PNG")
    return 224, 224, 3, path.stat().st_size


@router.post("/seed-demo")
def seed_demo(
    x_seed_secret: str = Header(..., alias="X-Seed-Secret"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    expected = os.getenv("SEED_SECRET")
    if not expected:
        raise HTTPException(500, "SEED_SECRET not configured on server")
    if x_seed_secret != expected:
        raise HTTPException(403, "Invalid seed secret")

    upload_dir = settings.UPLOAD_DIR / str(user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    created_images = []
    for i in range(NUM_IMAGES):
        label = BREAST_LABELS[i % 2]
        filename = f"{uuid.uuid4().hex}.png"
        dest = upload_dir / filename
        w, h, ch, size = _make_synthetic_image(dest, label)
        record = create_image(
            db,
            user_id=user.id,
            filename=filename,
            original_path=str(dest),
            image_type=ImageType.breast,
            width=w,
            height=h,
            channels=ch,
            file_size=size,
        )
        created_images.append((record, label))

    created_predictions: list[Prediction] = []
    for image, label in created_images:
        confidence = round(random.uniform(0.72, 0.97), 4)
        pred = create_prediction(
            db,
            image_id=image.id,
            model_name=MODEL_NAME,
            prediction_label=label,
            confidence=confidence,
            processing_time_ms=random.randint(140, 380),
        )
        created_predictions.append(pred)

    created_predictions.sort(key=lambda p: p.confidence, reverse=True)
    report_count = 0
    report_errors: list[str] = []
    for pred in created_predictions[:NUM_REPORTS]:
        try:
            from reports.generator import generate_report

            pdf_path = generate_report(pred.id, db)
            create_report(db, prediction_id=pred.id, pdf_path=pdf_path)
            report_count += 1
        except Exception as e:
            report_errors.append(f"prediction {pred.id}: {e!s}")

    return {
        "user": user.username,
        "images_created": len(created_images),
        "predictions_created": len(created_predictions),
        "reports_created": report_count,
        "report_errors": report_errors,
    }


@router.delete("/seed-demo")
def wipe_demo(
    x_seed_secret: str = Header(..., alias="X-Seed-Secret"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete all images/predictions/reports for the current user."""
    expected = os.getenv("SEED_SECRET")
    if not expected:
        raise HTTPException(500, "SEED_SECRET not configured on server")
    if x_seed_secret != expected:
        raise HTTPException(403, "Invalid seed secret")

    image_ids = [i.id for i in db.query(Image).filter(Image.user_id == user.id).all()]
    pred_ids = [
        p.id
        for p in db.query(Prediction).filter(Prediction.image_id.in_(image_ids)).all()
    ]
    n_reports = (
        db.query(Report).filter(Report.prediction_id.in_(pred_ids)).delete(synchronize_session=False)
    )
    n_preds = (
        db.query(Prediction).filter(Prediction.image_id.in_(image_ids)).delete(synchronize_session=False)
    )
    n_images = (
        db.query(Image).filter(Image.user_id == user.id).delete(synchronize_session=False)
    )
    db.commit()

    return {
        "user": user.username,
        "images_deleted": n_images,
        "predictions_deleted": n_preds,
        "reports_deleted": n_reports,
    }
