from fastapi import APIRouter

from .auth import router as auth_router
from .images import router as images_router
from .predict import router as predict_router
from .security import router as security_router
from .processing import router as processing_router
from .metrics import router as metrics_router
from .reports import router as reports_router
from .training import router as training_router
from .xai import router as xai_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(images_router, prefix="/images", tags=["Images"])
api_router.include_router(predict_router, prefix="/predict", tags=["Prediction"])
api_router.include_router(security_router, prefix="/security", tags=["Security"])
api_router.include_router(processing_router, prefix="/processing", tags=["Processing"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(reports_router, prefix="/reports", tags=["Reports"])
api_router.include_router(training_router, prefix="/training", tags=["Training"])
api_router.include_router(xai_router, prefix="/xai", tags=["XAI"])
