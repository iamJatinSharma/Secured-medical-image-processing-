import os
import uuid
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from db.database import get_db
from db.models import Image, Prediction, Report, User
from db.schemas import ImageResponse
from db.crud import create_image, get_images_by_user
from api.deps import get_current_user
from config.settings import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "module": "images"}


@router.post("/upload", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    image_type: str = Form("breast"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Ensure upload dir exists
    upload_dir = settings.UPLOAD_DIR / str(current_user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    ext = Path(file.filename or "image.png").suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = upload_dir / filename

    content = await file.read()
    filepath.write_bytes(content)

    # Get image dimensions
    width, height, channels = None, None, None
    try:
        from PIL import Image as PILImage
        img = PILImage.open(filepath)
        width, height = img.size
        channels = len(img.getbands())
    except Exception:
        pass

    image_record = create_image(
        db,
        user_id=current_user.id,
        filename=filename,
        original_path=str(filepath),
        image_type=image_type,
        width=width,
        height=height,
        channels=channels,
        file_size=len(content),
    )
    return image_record


@router.get("/", response_model=list[ImageResponse])
async def list_images(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_images_by_user(db, current_user.id)


@router.get("/stats")
async def image_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total_images = db.query(func.count(Image.id)).filter(Image.user_id == current_user.id).scalar() or 0
    total_predictions = db.query(func.count(Prediction.id)).join(Image).filter(Image.user_id == current_user.id).scalar() or 0
    total_reports = db.query(func.count(Report.id)).join(Prediction).join(Image).filter(Image.user_id == current_user.id).scalar() or 0

    recent_predictions = (
        db.query(Prediction)
        .join(Image)
        .filter(Image.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "total_images": total_images,
        "total_predictions": total_predictions,
        "total_reports": total_reports,
        "recent_predictions": [
            {
                "id": p.id,
                "image_id": p.image_id,
                "model_name": p.model_name,
                "prediction_label": p.prediction_label,
                "confidence": p.confidence,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "processing_time_ms": p.processing_time_ms,
            }
            for p in recent_predictions
        ],
    }
