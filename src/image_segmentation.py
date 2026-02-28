"""
Module: image_segmentation.py
Description: Functions for segmenting medical images (e.g., thresholding, region growing).
"""
import cv2
import numpy as np


def threshold_segmentation(image_path, output_path, thresh_val=128):
    image = cv2.imread(image_path, 0)
    _, segmented = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, segmented)
    print(f"Image segmented and saved to {output_path}")

if __name__ == "__main__":
    # threshold_segmentation('data/sample.png', 'results/segmented.png')
    pass
