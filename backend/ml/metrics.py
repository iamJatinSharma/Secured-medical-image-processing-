"""Evaluation metrics for image quality and model performance."""
import numpy as np
from typing import Optional


def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    return float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))


def calculate_psnr(original: np.ndarray, processed: np.ndarray, max_pixel: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio.
    
    Returns float('inf') for identical images.
    """
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_pixel ** 2) / mse))


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM).
    
    Simplified implementation without scipy dependency.
    Uses the standard SSIM formula with default constants.
    """
    img1 = original.astype(np.float64)
    img2 = processed.astype(np.float64)

    # If multi-channel, compute per-channel and average
    if img1.ndim == 3:
        channels = img1.shape[2]
        ssim_vals = []
        for c in range(channels):
            ssim_vals.append(_ssim_single_channel(img1[:, :, c], img2[:, :, c]))
        return float(np.mean(ssim_vals))
    else:
        return float(_ssim_single_channel(img1, img2))


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM for a single channel."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = _uniform_filter(img1, size=11)
    mu2 = _uniform_filter(img2, size=11)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _uniform_filter(img1 ** 2, size=11) - mu1_sq
    sigma2_sq = _uniform_filter(img2 ** 2, size=11) - mu2_sq
    sigma12 = _uniform_filter(img1 * img2, size=11) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def _uniform_filter(image: np.ndarray, size: int = 11) -> np.ndarray:
    """Simple uniform (box) filter using cumulative sum for speed."""
    pad = size // 2
    padded = np.pad(image, pad, mode="reflect")
    # Use cumsum for efficient box filter
    result = np.zeros_like(image)
    h, w = image.shape

    # Cumsum along rows
    cs = np.cumsum(padded, axis=0)
    row_sums = cs[size:, :] - cs[:-size, :]
    # Cumsum along cols
    cs2 = np.cumsum(row_sums, axis=1)
    box = cs2[:, size:] - cs2[:, :-size]

    # Trim to original size
    result = box[:h, :w] / (size * size)
    return result


def calculate_f1(y_true: list, y_pred: list, average: str = "binary") -> float:
    """Calculate F1 score."""
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ImportError:
        # Manual binary F1
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))


def calculate_auc(y_true: list, y_probs: list) -> float:
    """Calculate Area Under ROC Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_probs))
    except ImportError:
        # Simplified AUC via trapezoidal rule
        pairs = sorted(zip(y_probs, y_true), reverse=True)
        tp = 0
        fp = 0
        total_pos = sum(y_true)
        total_neg = len(y_true) - total_pos
        if total_pos == 0 or total_neg == 0:
            return 0.5
        auc = 0.0
        prev_fpr = 0.0
        for prob, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                fpr = fp / total_neg
                tpr = tp / total_pos
                auc += (fpr - prev_fpr) * tpr
                prev_fpr = fpr
        return float(auc)


def generate_confusion_matrix(y_true: list, y_pred: list, labels: Optional[list] = None) -> list:
    """Generate confusion matrix as 2D list."""
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.tolist()
    except ImportError:
        # Manual 2-class confusion matrix
        if labels is None:
            labels = sorted(set(y_true + y_pred))
        n = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        matrix = [[0] * n for _ in range(n)]
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                matrix[label_to_idx[t]][label_to_idx[p]] += 1
        return matrix


def generate_classification_report(y_true: list, y_pred: list, labels: Optional[list] = None) -> dict:
    """Generate classification report as dict."""
    try:
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    except ImportError:
        # Simplified report
        if labels is None:
            labels = sorted(set(y_true + y_pred))
        report = {}
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            report[str(label)] = {"precision": precision, "recall": recall, "f1-score": f1, "support": tp + fn}
        return report


def generate_roc_data(y_true: list, y_probs: list) -> dict:
    """Generate ROC curve data points."""
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc),
        }
    except ImportError:
        # Simplified ROC
        pairs = sorted(zip(y_probs, y_true), reverse=True)
        total_pos = sum(y_true)
        total_neg = len(y_true) - total_pos
        fpr_list = [0.0]
        tpr_list = [0.0]
        tp = 0
        fp = 0
        for prob, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            fpr_list.append(fp / total_neg if total_neg > 0 else 0)
            tpr_list.append(tp / total_pos if total_pos > 0 else 0)
        return {"fpr": fpr_list, "tpr": tpr_list, "thresholds": [], "auc": calculate_auc(y_true, y_probs)}
