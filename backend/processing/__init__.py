"""Image processing pipeline — denoise, enhance, segment."""
from processing.denoise import denoise_image
from processing.enhance import enhance_image
from processing.segment import segment_image


def process_image(image, steps: list[dict]) -> tuple[list[dict], object]:
    """Run a pipeline of processing steps on an image.
    
    Args:
        image: numpy array (H, W, 3) uint8
        steps: list of dicts, each with 'type' ('denoise'/'enhance'/'segment') and 'method'
    
    Returns:
        tuple of (list of result dicts, final processed image)
    """
    results = []
    current_image = image.copy()

    for step in steps:
        step_type = step.get("type", "")
        method = step.get("method", "")

        if step_type == "denoise":
            result = denoise_image(current_image, method=method)
            current_image = result["image"]
        elif step_type == "enhance":
            result = enhance_image(current_image, method=method)
            current_image = result["image"]
        elif step_type == "segment":
            result = segment_image(current_image, method=method)
        else:
            result = {"error": f"Unknown step type: {step_type}", "method": method, "time_ms": 0}

        results.append({
            "type": step_type,
            "method": method,
            "time_ms": result.get("time_ms", 0),
        })

    return results, current_image


__all__ = ["denoise_image", "enhance_image", "segment_image", "process_image"]
