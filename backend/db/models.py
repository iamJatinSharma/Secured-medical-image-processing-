import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from .database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    doctor = "doctor"
    researcher = "researcher"
    viewer = "viewer"


class ImageType(str, enum.Enum):
    breast = "breast"
    lung = "lung"
    brain = "brain"
    skin = "skin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.viewer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    images = relationship("Image", back_populates="user")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(512), nullable=False)
    processed_path = Column(String(512), nullable=True)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    image_type = Column(SAEnum(ImageType), nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)

    user = relationship("User", back_populates="images")
    predictions = relationship("Prediction", back_populates="image")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    prediction_label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    gradcam_path = Column(String(512), nullable=True)
    lime_path = Column(String(512), nullable=True)
    shap_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_time_ms = Column(Integer, nullable=True)

    image = relationship("Image", back_populates="predictions")
    reports = relationship("Report", back_populates="prediction")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    pdf_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="reports")
