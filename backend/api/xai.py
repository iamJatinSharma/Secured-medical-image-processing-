"""XAI (Explainable AI) API endpoints — Grad-CAM, LIME, SHAP."""
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
    return {"status": "ok", "module": "xai"}


@router.post("/gradcam")
async def generate_gradcam(
    image: UploadFile = File(...),
    model_name: str = Form("resnet50_breast"),
    target_class: int = Form(None),
    current_user: User = Depends(get_current_user),
):
    """Generate Grad-CAM heatmap for an image."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from xai.gradcam import GradCAM, overlay_heatmap
        from ml.model_registry import get_model_and_config

        model, config = get_model_and_config(model_name)
        target_layer = model.layer4[-1] if hasattr(model, "layer4") else None
        if target_layer is None:
            raise HTTPException(status_code=400, detail="Model architecture not supported for Grad-CAM")

        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(img_array, target_class=target_class)
        overlay = overlay_heatmap(img_array, heatmap)

        # Return overlay image
        out_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"XAI module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


@router.post("/gradcam/heatmap")
async def generate_gradcam_raw(
    image: UploadFile = File(...),
    model_name: str = Form("resnet50_breast"),
    target_class: int = Form(None),
    current_user: User = Depends(get_current_user),
):
    """Generate raw Grad-CAM heatmap (grayscale) without overlay."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from xai.gradcam import GradCAM
        from ml.model_registry import get_model_and_config

        model, config = get_model_and_config(model_name)
        target_layer = model.layer4[-1] if hasattr(model, "layer4") else None
        if target_layer is None:
            raise HTTPException(status_code=400, detail="Model not supported for Grad-CAM")

        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(img_array, target_class=target_class)

        # Return as grayscale PNG
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        out_img = Image.fromarray(heatmap_uint8, mode="L")
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"XAI module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


@router.post("/lime")
async def generate_lime(
    image: UploadFile = File(...),
    model_name: str = Form("resnet50_breast"),
    num_samples: int = Form(100),
    current_user: User = Depends(get_current_user),
):
    """Generate LIME explanation for an image."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from xai.lime_explainer import generate_lime_explanation
        from ml.model_registry import get_model_and_config

        model, config = get_model_and_config(model_name)
        explanation_img = generate_lime_explanation(model, img_array, num_samples=num_samples)

        out_img = Image.fromarray(explanation_img)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"LIME module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME generation failed: {e}")


@router.post("/shap")
async def generate_shap(
    image: UploadFile = File(...),
    model_name: str = Form("resnet50_breast"),
    current_user: User = Depends(get_current_user),
):
    """Generate SHAP explanation for an image."""
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    try:
        from xai.shap_explainer import generate_shap_explanation
        from ml.model_registry import get_model_and_config

        model, config = get_model_and_config(model_name)
        shap_img = generate_shap_explanation(model, img_array)

        if shap_img is not None:
            out_img = Image.fromarray(shap_img)
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="SHAP generation returned no result")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"SHAP module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP generation failed: {e}")


@router.get("/methods")
async def list_xai_methods():
    """List available XAI methods."""
    return {
        "methods": [
            {
                "id": "gradcam",
                "name": "Grad-CAM",
                "description": "Gradient-weighted Class Activation Mapping — highlights important regions using gradients",
                "supported_models": ["resnet50_breast", "resnet50_lung", "resnet50_brain", "resnet50_skin"],
            },
            {
                "id": "lime",
                "name": "LIME",
                "description": "Local Interpretable Model-agnostic Explanations — identifies important superpixels",
                "supported_models": ["resnet50_breast", "resnet50_lung", "resnet50_brain", "resnet50_skin"],
            },
            {
                "id": "shap",
                "name": "SHAP",
                "description": "SHapley Additive exPlanations — game theory based feature attribution",
                "supported_models": ["resnet50_breast", "resnet50_lung", "resnet50_brain", "resnet50_skin"],
            },
        ]
    }
