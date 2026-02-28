"""
Module: image_denoising.py
Description: Functions for denoising medical images.
"""
import cv2


def denoise_image(image_path, output_path):
    image = cv2.imread(image_path)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    cv2.imwrite(output_path, denoised)
    print(f"Image denoised and saved to {output_path}")

if __name__ == "__main__":
    # denoise_image('data/sample.png', 'results/denoised.png')
    pass
