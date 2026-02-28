"""
Main Application: Secure Medical Image Processing
Run this file to execute the complete workflow.
"""
import os
import sys

from evaluation_metrics import calculate_psnr, calculate_ssim
from image_denoising import denoise_image
from image_optimization import compress_image, enhance_image
from image_security import decrypt_image, encrypt_image, generate_key
from image_segmentation import threshold_segmentation
from user_authentication import hash_password, verify_password
from watermarking import add_watermark

# Create results folder if not exists
os.makedirs('../results', exist_ok=True)

def main():
    print("=" * 60)
    print("SECURE MEDICAL IMAGE PROCESSING SYSTEM")
    print("=" * 60)
    
    # Step 1: User Authentication
    print("\n[Step 1] User Authentication")
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    # Hash password (in real app, compare with stored hash)
    hashed = hash_password(password)
    if verify_password(hashed, password):
        print("✓ Authentication successful!")
    else:
        print("✗ Authentication failed!")
        return
    
    # Step 2: Select an image
    print("\n[Step 2] Select Image")
    image_path = input("Enter path to medical image (or press Enter for default): ").strip()
    if not image_path:
        # Check if data folder has images (search recursively)
        data_folder = '../data'
        if os.path.exists(data_folder):
            image_path = None
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        break
                if image_path:
                    break
            
            if image_path:
                print(f"Using default image: {image_path}")
            else:
                print("No images found in data folder. Please add images and try again.")
                return
        else:
            print("Data folder not found. Please add images to the data folder.")
            return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Step 3: Image Denoising
    print("\n[Step 3] Image Denoising")
    denoised_path = '../results/denoised.png'
    denoise_image(image_path, denoised_path)
    
    # Step 4: Image Enhancement
    print("\n[Step 4] Image Enhancement")
    enhanced_path = '../results/enhanced.png'
    enhance_image(denoised_path, enhanced_path)
    
    # Step 5: Image Segmentation
    print("\n[Step 5] Image Segmentation")
    segmented_path = '../results/segmented.png'
    threshold_segmentation(enhanced_path, segmented_path)
    
    # Step 6: Add Watermark
    print("\n[Step 6] Adding Watermark")
    watermarked_path = '../results/watermarked.png'
    add_watermark(enhanced_path, 'CONFIDENTIAL', watermarked_path)
    
    # Step 7: Compress Image
    print("\n[Step 7] Image Compression")
    compressed_path = '../results/compressed.jpg'
    compress_image(watermarked_path, compressed_path, quality=85)
    
    # Step 8: Encrypt Image
    print("\n[Step 8] Image Encryption")
    key_path = '../results/secret.key'
    if not os.path.exists(key_path):
        print("Generating encryption key...")
        generate_key(key_path)
    encrypted_path = '../results/encrypted.img'
    encrypt_image(watermarked_path, key_path, encrypted_path)
    
    # Step 9: Evaluation Metrics
    print("\n[Step 9] Evaluation Metrics")
    try:
        psnr = calculate_psnr(image_path, denoised_path)
        print(f"PSNR (Original vs Denoised): {psnr:.2f} dB")
        
        ssim = calculate_ssim(image_path, denoised_path)
        print(f"SSIM (Original vs Denoised): {ssim:.4f}")
    except Exception as e:
        print(f"Could not calculate metrics: {e}")
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print("Results saved in 'results/' folder:")
    print("  - denoised.png (noise removed)")
    print("  - enhanced.png (enhanced quality)")
    print("  - segmented.png (segmented regions)")
    print("  - watermarked.png (with watermark)")
    print("  - compressed.jpg (optimized size)")
    print("  - encrypted.img (encrypted, secure)")
    print("  - secret.key (encryption key - keep safe!)")
    print("\nNote: For disease detection, train an ML model first.")
    print("=" * 60)

if __name__ == "__main__":
    main()
