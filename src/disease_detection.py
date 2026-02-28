"""
Module: disease_detection.py
Description: ML-based disease detection from medical images (e.g., cancer detection using a simple CNN).
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def predict_disease(image_path, model_path, class_names):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    model = load_model(model_path)
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"Predicted disease: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    # Example usage:
    # class_names = ['Normal', 'Cancer']
    # predict_disease('data/sample.png', 'models/cancer_model.h5', class_names)
    pass
