"""
Module: test_model.py
Description: Test the trained model on new images and display predictions.
"""
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = '../models/cancer_detection_model.h5'
IMG_SIZE = 50

def predict_single_image(image_path, model_path=MODEL_PATH):
    """
    Predict cancer in a single image.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return None
    
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img)[0][0]
    
    if prediction > 0.5:
        result = "Cancer Detected"
        confidence = prediction * 100
    else:
        result = "No Cancer"
        confidence = (1 - prediction) * 100
    
    print(f"\nImage: {image_path}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    
    return result, confidence

def test_multiple_images(data_path='../data/breast-histopathology-images'):
    """
    Test the model on multiple images from the dataset.
    """
    print("=" * 60)
    print("TESTING CANCER DETECTION MODEL")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Please train the model first by running: python train_model.py")
        return
    
    model = load_model(MODEL_PATH)
    
    # Get some test images
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()]
    
    correct = 0
    total = 0
    
    for patient_id in patient_dirs[:5]:  # Test first 5 patients
        patient_path = os.path.join(data_path, patient_id)
        
        for class_label in ['0', '1']:
            class_path = os.path.join(patient_path, class_label)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith('.png')][:2]  # 2 images per class
                
                for img_file in images:
                    img_path = os.path.join(class_path, img_file)
                    
                    # Load and predict
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    prediction = model.predict(img, verbose=0)[0][0]
                    predicted_class = 1 if prediction > 0.5 else 0
                    actual_class = int(class_label)
                    
                    total += 1
                    if predicted_class == actual_class:
                        correct += 1
                        status = "✓"
                    else:
                        status = "✗"
                    
                    print(f"{status} Image: {img_file} | Actual: {actual_class} | Predicted: {predicted_class} | Confidence: {prediction:.2f}")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n" + "=" * 60)
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    test_multiple_images()
