"""Image enhancement methods using OpenCV."""
import numpy as np
import time

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def enhance_image(image: np.ndarray, method: str = "clahe", **kwargs) -> dict:
    """Enhance image using specified method.
    
    Args:
        image: Input image as numpy array (H, W, 3) uint8
        method: One of 'clahe', 'detail', 'histogram'
    
    Returns:
        dict with 'image' (enhanced array), 'method', 'time_ms'
    """
    start = time.time()

    if not HAS_CV2:
        return {
            "image": image.copy(),
            "method": method,
            "time_ms": 0,
            "note": "cv2 not available, returning original",
        }

    if method == "clahe":
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_size = kwargs.get("tile_size", 8)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
        )
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    elif method == "detail":
        sigma_s = kwargs.get("sigma_s", 10)
        sigma_r = kwargs.get("sigma_r", 0.15)
        result = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
    elif method == "histogram":
        # Per-channel histogram equalization
        channels = cv2.split(image)
        equalized = [cv2.equalizeHist(ch) for ch in channels]
        result = cv2.merge(equalized)
    else:
        raise ValueError(f"Unknown enhance method: {method}. Use 'clahe', 'detail', or 'histogram'.")

    elapsed = (time.time() - start) * 1000
    return {"image": result, "method": method, "time_ms": round(elapsed, 2)}
