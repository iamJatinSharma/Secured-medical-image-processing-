"""Security API endpoints — encryption and watermarking."""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
import numpy as np
from PIL import Image
import io

from db.database import get_db
from api.deps import get_current_user
from db.models import User

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "module": "security"}


@router.post("/encrypt")
async def encrypt_image_endpoint(
    image: UploadFile = File(...),
    method: str = Form("arnold"),
    password: str = Form(""),
    iterations: int = Form(20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Encrypt an image using specified method (arnold/logistic/aes)."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        if method == "arnold":
            from security.chaos_encrypt import arnold_cat_map
            encrypted = arnold_cat_map(img_array, iterations=iterations)
            # Return as PNG
            out_img = Image.fromarray(encrypted)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={
                    "X-Original-Width": str(img_array.shape[1]),
                    "X-Original-Height": str(img_array.shape[0]),
                    "X-Method": "arnold",
                    "X-Iterations": str(iterations),
                },
            )
        elif method == "logistic":
            from security.logistic_encrypt import logistic_map_encrypt
            encrypted = logistic_map_encrypt(img_array)
            out_img = Image.fromarray(encrypted)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={"X-Method": "logistic"},
            )
        elif method == "aes":
            from security.aes_encrypt import aes_encrypt
            if not password:
                raise HTTPException(status_code=400, detail="Password required for AES encryption")
            img_bytes = img_array.tobytes()
            encrypted_bytes = aes_encrypt(img_bytes, password)
            return Response(
                content=encrypted_bytes,
                media_type="application/octet-stream",
                headers={
                    "X-Original-Width": str(img_array.shape[1]),
                    "X-Original-Height": str(img_array.shape[0]),
                    "X-Original-Channels": str(img_array.shape[2]),
                    "X-Method": "aes",
                },
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Encryption module not available: {e}")


@router.post("/decrypt")
async def decrypt_image_endpoint(
    file: UploadFile = File(...),
    method: str = Form("arnold"),
    password: str = Form(""),
    iterations: int = Form(20),
    width: int = Form(0),
    height: int = Form(0),
    channels: int = Form(3),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Decrypt an encrypted image."""
    contents = await file.read()

    try:
        if method == "arnold":
            from security.chaos_encrypt import arnold_cat_map_decrypt
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = np.array(img)
            decrypted = arnold_cat_map_decrypt(img_array, iterations=iterations, original_shape=img_array.shape)
            out_img = Image.fromarray(decrypted)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        elif method == "logistic":
            from security.logistic_encrypt import logistic_map_decrypt
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = np.array(img)
            decrypted = logistic_map_decrypt(img_array)
            out_img = Image.fromarray(decrypted)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        elif method == "aes":
            from security.aes_encrypt import aes_decrypt
            if not password:
                raise HTTPException(status_code=400, detail="Password required for AES decryption")
            decrypted_bytes = aes_decrypt(contents, password)
            if width == 0 or height == 0:
                raise HTTPException(status_code=400, detail="Width and height required for AES decryption")
            img_array = np.frombuffer(decrypted_bytes, dtype=np.uint8).reshape(height, width, channels)
            out_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    except Exception as e:
        if "InvalidTag" in str(type(e).__name__) or "tag" in str(e).lower():
            raise HTTPException(status_code=401, detail="Invalid password")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watermark/embed")
async def watermark_embed(
    image: UploadFile = File(...),
    text: str = Form(...),
    method: str = Form("dct"),
    current_user: User = Depends(get_current_user),
):
    """Embed invisible or visible watermark into image."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        if method == "visible":
            from security.watermark import embed_visible_watermark
            result = embed_visible_watermark(img_array, text)
        else:
            from security.watermark import embed_watermark
            result = embed_watermark(img_array, text, method=method)

        out_img = Image.fromarray(result)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Watermark module not available: {e}")


@router.post("/watermark/extract")
async def watermark_extract(
    image: UploadFile = File(...),
    method: str = Form("dct"),
    current_user: User = Depends(get_current_user),
):
    """Extract watermark text from image."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from security.watermark import extract_watermark
        text = extract_watermark(img_array, method=method)
        return {"extracted_text": text, "method": method}
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Watermark module not available: {e}")


@router.post("/watermark/verify")
async def watermark_verify(
    image: UploadFile = File(...),
    expected_text: str = Form(...),
    method: str = Form("dct"),
    current_user: User = Depends(get_current_user),
):
    """Verify if image contains expected watermark."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from security.watermark import verify_watermark
        is_valid = verify_watermark(img_array, expected_text, method=method)
        return {"verified": is_valid, "expected_text": expected_text}
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Watermark module not available: {e}")


@router.post("/compare")
async def compare_images(
    original: UploadFile = File(...),
    processed: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Compare two images — returns PSNR and visual difference."""
    orig_contents = await original.read()
    proc_contents = await processed.read()

    orig_img = np.array(Image.open(io.BytesIO(orig_contents)).convert("RGB"))
    proc_img = np.array(Image.open(io.BytesIO(proc_contents)).convert("RGB"))

    if orig_img.shape != proc_img.shape:
        raise HTTPException(status_code=400, detail="Images must have same dimensions")

    mse = float(np.mean((orig_img.astype(float) - proc_img.astype(float)) ** 2))
    psnr = float(10 * np.log10(255**2 / mse)) if mse > 0 else float("inf")

    return {"psnr_db": round(psnr, 2), "mse": round(mse, 4), "identical": mse == 0}
