"""Image denoising methods using OpenCV."""
import numpy as np
import time

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def denoise_image(image: np.ndarray, method: str = "nlm", **kwargs) -> dict:
    """Denoise image using specified method.
    
    Args:
        image: Input image as numpy array (H, W, 3) uint8
        method: One of 'nlm', 'bilateral', 'gaussian'
    
    Returns:
        dict with 'image' (denoised array), 'method', 'time_ms'
    """
    start = time.time()

    if not HAS_CV2:
        return {
            "image": image.copy(),
            "method": method,
            "time_ms": 0,
            "note": "cv2 not available, returning original",
        }

    if method == "nlm":
        h = kwargs.get("h", 10)
        template_window = kwargs.get("template_window", 7)
        search_window = kwargs.get("search_window", 21)
        result = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window, search_window
        )
    elif method == "bilateral":
        d = kwargs.get("d", 9)
        sigma_color = kwargs.get("sigma_color", 75)
        sigma_space = kwargs.get("sigma_space", 75)
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    elif method == "gaussian":
        ksize = kwargs.get("ksize", 5)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(image, (ksize, ksize), 0)
    else:
        raise ValueError(f"Unknown denoise method: {method}. Use 'nlm', 'bilateral', or 'gaussian'.")

    elapsed = (time.time() - start) * 1000
    return {"image": result, "method": method, "time_ms": round(elapsed, 2)}
