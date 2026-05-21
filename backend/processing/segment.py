"""Image segmentation methods using OpenCV."""
import numpy as np
import time

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def segment_image(image: np.ndarray, method: str = "threshold", **kwargs) -> dict:
    """Segment image using specified method.
    
    Args:
        image: Input image as numpy array (H, W, 3) uint8
        method: One of 'threshold', 'watershed', 'kmeans'
    
    Returns:
        dict with 'mask' (segmentation mask), 'overlay' (colored overlay), 'method', 'time_ms'
    """
    start = time.time()

    if not HAS_CV2:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        overlay = image.copy()
        return {
            "mask": mask,
            "overlay": overlay,
            "method": method,
            "time_ms": 0,
            "note": "cv2 not available",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "threshold":
        thresh_val = kwargs.get("threshold", 0)
        if thresh_val == 0:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    elif method == "watershed":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Morphological operations to find sure background/foreground
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255

    elif method == "kmeans":
        k = kwargs.get("k", 3)
        pixel_data = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape(image.shape[:2])
        # Use the darkest cluster as background
        center_brightness = centers.mean(axis=1)
        bg_label = np.argmin(center_brightness)
        mask = np.where(labels != bg_label, 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown segment method: {method}. Use 'threshold', 'watershed', or 'kmeans'.")

    # Create colored overlay
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green for segmented regions
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    elapsed = (time.time() - start) * 1000
    return {"mask": mask, "overlay": overlay, "method": method, "time_ms": round(elapsed, 2)}
