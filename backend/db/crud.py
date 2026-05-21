from sqlalchemy.orm import Session
from passlib.context import CryptContext

from .models import User, Image, Prediction, Report, ImageType
from .schemas import UserCreate

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- User CRUD ---

def create_user(db: Session, user_data: UserCreate) -> User:
    hashed_password = pwd_context.hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# --- Image CRUD ---

def create_image(
    db: Session,
    user_id: int,
    filename: str,
    original_path: str,
    image_type: ImageType,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
    file_size: int | None = None,
) -> Image:
    image = Image(
        user_id=user_id,
        filename=filename,
        original_path=original_path,
        image_type=image_type,
        width=width,
        height=height,
        channels=channels,
        file_size=file_size,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


def get_images_by_user(db: Session, user_id: int) -> list[Image]:
    return db.query(Image).filter(Image.user_id == user_id).all()


# --- Prediction CRUD ---

def create_prediction(
    db: Session,
    image_id: int,
    model_name: str,
    prediction_label: str,
    confidence: float,
    processing_time_ms: int | None = None,
    gradcam_path: str | None = None,
    lime_path: str | None = None,
    shap_path: str | None = None,
) -> Prediction:
    prediction = Prediction(
        image_id=image_id,
        model_name=model_name,
        prediction_label=prediction_label,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        gradcam_path=gradcam_path,
        lime_path=lime_path,
        shap_path=shap_path,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


# --- Report CRUD ---

def create_report(
    db: Session,
    prediction_id: int,
    pdf_path: str,
) -> Report:
    report = Report(
        prediction_id=prediction_id,
        pdf_path=pdf_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
