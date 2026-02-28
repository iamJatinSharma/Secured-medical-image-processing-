"""
Module: image_optimization.py
Description: Functions for image compression and enhancement.
"""
import cv2
import numpy as np


def compress_image(image_path, output_path, quality=90):
    image = cv2.imread(image_path)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Image compressed and saved to {output_path}")

def enhance_image(image_path, output_path):
    image = cv2.imread(image_path)
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    cv2.imwrite(output_path, enhanced)
    print(f"Image enhanced and saved to {output_path}")

if __name__ == "__main__":
    # compress_image('data/sample.png', 'results/compressed.jpg')
    # enhance_image('data/sample.png', 'results/enhanced.png')
    pass
