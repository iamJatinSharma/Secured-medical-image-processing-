import time

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Image, User
from db.crud import create_prediction
from db.schemas import PredictionResponse
from api.deps import get_current_user
from ml.model_registry import predict, get_available_models, get_model_config

router = APIRouter()


class DetectRequest(BaseModel):
    image_id: int
    model_name: str


class DetectResponse(BaseModel):
    id: int
    image_id: int
    model_name: str
    prediction_label: str
    confidence: float
    all_probabilities: dict[str, float]
    processing_time_ms: int | None


@router.get("/health")
async def health():
    return {"status": "ok", "module": "predict"}


@router.get("/models")
async def list_models():
    """Return all available detection models."""
    return get_available_models()


@router.get("/predictions")
async def list_predictions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List predictions owned by current user (most recent first)."""
    from db.models import Prediction as P
    rows = (
        db.query(P)
        .join(Image, P.image_id == Image.id)
        .filter(Image.user_id == current_user.id)
        .order_by(P.created_at.desc())
        .all()
    )
    return [
        {
            "id": p.id,
            "image_id": p.image_id,
            "model_name": p.model_name,
            "prediction_label": p.prediction_label,
            "confidence": p.confidence,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "processing_time_ms": p.processing_time_ms,
        }
        for p in rows
    ]


@router.post("/detect", response_model=DetectResponse)
async def detect(
    req: DetectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate model exists
    config = get_model_config(req.model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")

    # Validate image exists and belongs to user
    image = db.query(Image).filter(Image.id == req.image_id, Image.user_id == current_user.id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Run prediction
    try:
        result = predict(req.model_name, image.original_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Save to DB
    prediction = create_prediction(
        db,
        image_id=image.id,
        model_name=req.model_name,
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        processing_time_ms=result["processing_time_ms"],
    )

    return DetectResponse(
        id=prediction.id,
        image_id=prediction.image_id,
        model_name=prediction.model_name,
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        processing_time_ms=result["processing_time_ms"],
    )
