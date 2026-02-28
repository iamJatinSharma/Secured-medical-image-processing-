"""
Module: evaluation_metrics.py
Description: Functions to evaluate image quality and ML model performance.
"""
import cv2
import numpy as np
from sklearn.metrics import accuracy_score


def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def calculate_ssim(img1_path, img2_path):
    from skimage.metrics import structural_similarity as ssim
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate appropriate window size (must be odd and <= smallest dimension)
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    
    # Ensure win_size is at least 3 and odd
    if win_size < 3:
        win_size = 3
    
    # Convert to grayscale if multichannel
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(img1_gray, img2_gray, win_size=win_size, data_range=255)
    else:
        ssim_value = ssim(img1, img2, win_size=win_size, data_range=255)
    
    return ssim_value

def evaluate_model(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    # print(calculate_psnr('results/denoised.png', 'data/sample.png'))
    # print(calculate_ssim('results/denoised.png', 'data/sample.png'))
    # print(evaluate_model([0,1,1], [0,1,0]))
    pass
