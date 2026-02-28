"""
Module: watermarking.py
Description: Functions for adding and detecting watermarks in medical images.
"""
import cv2
import numpy as np


def add_watermark(image_path, watermark_text, output_path):
    image = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, watermark_text, (10, image.shape[0] - 10), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, image)
    print(f"Watermark added and saved to {output_path}")

if __name__ == "__main__":
    # add_watermark('data/sample.png', 'Confidential', 'results/watermarked.png')
    pass
