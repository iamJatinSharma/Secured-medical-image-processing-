"""Seed demo data for a user: sample images, predictions, and a report.

Usage:
    python seed_demo.py [username]

If no username is given, seeds for the user with the lowest id. Pulls sample
patches from ../data/<patient>/{0,1}/ (Kaggle breast histopathology set),
copies them into the user's upload dir, runs predictions with whatever
breast model is registered, and generates one PDF report.
"""
from __future__ import annotations

import random
import shutil
import sys
import uuid
from pathlib import Path

from PIL import Image as PILImage

# Set CWD-aware imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db.database import SessionLocal  # noqa: E402
from db.models import Image, User  # noqa: E402
from db.crud import create_image, create_prediction  # noqa: E402
from config.settings import settings  # noqa: E402
from ml.model_registry import predict, get_all_configs  # noqa: E402
from reports.generator import generate_report  # noqa: E402

DATA_ROOT = ROOT.parent / "data"  # ../data
NUM_IMAGES = 8
NUM_REPORTS = 2


def pick_sample_images(n: int) -> list[Path]:
    """Pull n PNGs from the breast-histopathology dataset, mixing classes 0 and 1."""
    if not DATA_ROOT.exists():
        print(f"  ! Data folder not found at {DATA_ROOT}")
        return []

    patients = [p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    if not patients:
        print(f"  ! No patient folders in {DATA_ROOT}")
        return []

    picked: list[Path] = []
    # Try to get a mix of class 0 (no cancer) and class 1 (cancer)
    rng = random.Random(42)
    for cls in ("0", "1"):
        for _ in range(n // 2 + 1):
            patient = rng.choice(patients)
            cls_dir = patient / cls
            if not cls_dir.exists():
                continue
            files = list(cls_dir.glob("*.png"))
            if files:
                picked.append(rng.choice(files))
                if len(picked) >= n:
                    return picked
    return picked[:n]


def pick_breast_model() -> str:
    """Pick a trained model targeting breast tissue, or fall back to resnet50_breast."""
    configs = get_all_configs()
    # Prefer a trained MedMNIST breast model if present
    for name, cfg in configs.items():
        if cfg.get("trained") and ("breast" in name.lower() or cfg.get("dataset") == "breastmnist"):
            return name
    return "resnet50_breast"


def seed_user(username: str | None) -> None:
    db = SessionLocal()
    try:
        if username:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                print(f"  ! User '{username}' not found.")
                return
        else:
            user = db.query(User).order_by(User.id.asc()).first()
            if not user:
                print("  ! No users in DB.")
                return

        print(f"-> Seeding demo data for user: {user.username} (id={user.id})")

        # Where uploaded images live
        upload_dir = settings.UPLOAD_DIR / str(user.id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        sources = pick_sample_images(NUM_IMAGES)
        if not sources:
            print("  ! No source images found — aborting.")
            return

        print(f"  -> Copying {len(sources)} sample images")
        created_image_ids: list[int] = []
        for src in sources:
            ext = src.suffix
            unique_name = f"{uuid.uuid4().hex}{ext}"
            dest = upload_dir / unique_name
            shutil.copy(src, dest)

            try:
                im = PILImage.open(dest)
                w, h = im.size
                channels = len(im.getbands())
                file_size = dest.stat().st_size
            except Exception:
                w = h = channels = None
                file_size = None

            record = create_image(
                db,
                user_id=user.id,
                filename=unique_name,
                original_path=str(dest),
                image_type="breast",
                width=w,
                height=h,
                channels=channels,
                file_size=file_size,
            )
            created_image_ids.append(record.id)

        # Run a real prediction on each image
        model_name = pick_breast_model()
        print(f"  -> Running predictions with model: {model_name}")
        predictions = []
        for image_id in created_image_ids:
            img = db.query(Image).filter(Image.id == image_id).first()
            if not img:
                continue
            try:
                result = predict(model_name, img.original_path)
            except Exception as e:
                print(f"    ! predict failed for image {image_id}: {e}")
                continue
            pred = create_prediction(
                db,
                image_id=image_id,
                model_name=model_name,
                prediction_label=result["prediction_label"],
                confidence=result["confidence"],
                processing_time_ms=result.get("processing_time_ms"),
            )
            predictions.append(pred)
            print(f"    - image {image_id}: {result['prediction_label']} ({result['confidence']:.2f})")

        # Generate a few reports for highest-confidence predictions
        print(f"  -> Generating {min(NUM_REPORTS, len(predictions))} PDF reports")
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        for pred in predictions[:NUM_REPORTS]:
            try:
                pdf_path = generate_report(pred.id, db)
                print(f"    - report for prediction {pred.id}: {Path(pdf_path).name}")
            except Exception as e:
                print(f"    ! report failed for prediction {pred.id}: {e}")

        print("[OK] Done.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_user(sys.argv[1] if len(sys.argv) > 1 else None)
