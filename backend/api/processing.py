"""Image processing API endpoints."""
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import numpy as np
from PIL import Image
import io
from typing import Optional

from db.database import get_db
from api.deps import get_current_user
from db.models import User

router = APIRouter()


def _read_upload_image(file_bytes: bytes) -> np.ndarray:
    """Read uploaded file bytes into numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def _numpy_to_png_response(image: np.ndarray) -> StreamingResponse:
    """Convert numpy array to PNG streaming response."""
    img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/health")
async def health():
    return {"status": "ok", "module": "processing"}


@router.post("/denoise")
async def denoise(
    image: UploadFile = File(...),
    method: str = Form("nlm"),
    current_user: User = Depends(get_current_user),
):
    """Denoise an image. Methods: nlm, bilateral, gaussian."""
    from processing.denoise import denoise_image

    file_bytes = await image.read()
    img_array = _read_upload_image(file_bytes)
    result = denoise_image(img_array, method=method)
    return _numpy_to_png_response(result["image"])


@router.post("/enhance")
async def enhance(
    image: UploadFile = File(...),
    method: str = Form("clahe"),
    current_user: User = Depends(get_current_user),
):
    """Enhance an image. Methods: clahe, detail, histogram."""
    from processing.enhance import enhance_image

    file_bytes = await image.read()
    img_array = _read_upload_image(file_bytes)
    result = enhance_image(img_array, method=method)
    return _numpy_to_png_response(result["image"])


@router.post("/segment")
async def segment(
    image: UploadFile = File(...),
    method: str = Form("threshold"),
    current_user: User = Depends(get_current_user),
):
    """Segment an image. Methods: threshold, watershed, kmeans."""
    from processing.segment import segment_image

    file_bytes = await image.read()
    img_array = _read_upload_image(file_bytes)
    result = segment_image(img_array, method=method)
    return _numpy_to_png_response(result["overlay"])


@router.post("/pipeline")
async def pipeline(
    image: UploadFile = File(...),
    steps: str = Form('[]'),
    current_user: User = Depends(get_current_user),
):
    """Run a processing pipeline. Steps: JSON array of {type, method}."""
    import json
    from processing import process_image

    file_bytes = await image.read()
    img_array = _read_upload_image(file_bytes)

    try:
        step_list = json.loads(steps)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid steps JSON")

    results, final_image = process_image(img_array, step_list)

    # Return the final processed image
    return _numpy_to_png_response(final_image)


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

    # Resize to match if different sizes
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((w, h)))
        img2 = np.array(Image.fromarray(img2).resize((w, h)))

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    mse = calculate_mse(img1, img2)

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mse": mse,
        "identical": mse == 0,
    }
