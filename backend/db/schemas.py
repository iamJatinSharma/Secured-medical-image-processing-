from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- User Schemas ---

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


# --- Image Schemas ---

class ImageResponse(BaseModel):
    id: int
    user_id: int
    filename: str
    original_path: str
    processed_path: Optional[str] = None
    upload_time: datetime
    image_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    file_size: Optional[int] = None

    class Config:
        from_attributes = True


# --- Prediction Schemas ---

class PredictionResponse(BaseModel):
    id: int
    image_id: int
    model_name: str
    prediction_label: str
    confidence: float
    gradcam_path: Optional[str] = None
    lime_path: Optional[str] = None
    shap_path: Optional[str] = None
    created_at: datetime
    processing_time_ms: Optional[int] = None

    class Config:
        from_attributes = True


# --- Report Schemas ---

class ReportResponse(BaseModel):
    id: int
    prediction_id: int
    pdf_path: str
    created_at: datetime

    class Config:
        from_attributes = True
