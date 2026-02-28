"""
Module: train_model.py
Description: Train a CNN model for disease detection (breast cancer classification).
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
IMG_SIZE = 50  # Resize images to 50x50 (adjust based on your needs)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = '../models/cancer_detection_model.h5'

def load_dataset(data_path='../data/breast-histopathology-images'):
    """
    Load images from the dataset.
    Expected structure: data_path/patient_id/class/image.png
    Classes: 0 (no cancer), 1 (cancer - IDC)
    """
    images = []
    labels = []
    
    print("Loading dataset...")
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()]
    
    count = 0
    max_images = 5000  # Limit for faster training (remove for full dataset)
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(data_path, patient_id)
        
        # Class 0 (no cancer)
        class_0_path = os.path.join(patient_path, '0')
        if os.path.exists(class_0_path):
            for img_file in os.listdir(class_0_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_0_path, img_file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(0)
                    count += 1
                    if count >= max_images:
                        break
        
        if count >= max_images:
            break
        
        # Class 1 (cancer - IDC)
        class_1_path = os.path.join(patient_path, '1')
        if os.path.exists(class_1_path):
            for img_file in os.listdir(class_1_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_1_path, img_file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(1)
                    count += 1
                    if count >= max_images:
                        break
        
        if count >= max_images:
            break
        
        if count % 500 == 0:
            print(f"Loaded {count} images...")
    
    print(f"Total images loaded: {count}")
    print(f"Class 0 (No Cancer): {labels.count(0)}")
    print(f"Class 1 (Cancer): {labels.count(1)}")
    
    return np.array(images), np.array(labels)

def create_model():
    """
    Create a CNN model for binary classification.
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """
    Train the model.
    """
    print("=" * 60)
    print("TRAINING CANCER DETECTION MODEL")
    print("=" * 60)
    
    # Load dataset
    X, y = load_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer']))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n" + "=" * 60)
    print(f"Model saved to: {MODEL_PATH}")
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    train()
