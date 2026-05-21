"""
Module: disease_detection.py
Description: ML-based disease detection from medical images (CNN, ResNet, model evaluation).
"""

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from evaluation_metrics import calculate_auc, calculate_f1

# ──────────────────────────────────────────────
# 1. PREDICTION
# ──────────────────────────────────────────────

def predict_disease(image_path, model_path, class_names):
    """
    Load a saved model and predict the disease class for a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    model = load_model(model_path)
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    print(f"Predicted disease : {predicted_class}")
    print(f"Confidence        : {confidence:.2f}%")
    return predicted_class, confidence


# ──────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ──────────────────────────────────────────────

def build_resnet_model(num_classes, freeze_base=True):
    """
    Build a ResNet50-based transfer learning model.

    Args:
        num_classes  : Number of output disease classes.
        freeze_base  : If True, freeze all ResNet50 base layers (recommended for initial training).

    Returns:
        Compiled Keras Model.
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base layers — only the custom head will be trained initially
    base_model.trainable = not freeze_base

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)                      # Regularisation to prevent overfitting
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# ──────────────────────────────────────────────
# 3. EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(model, data_gen):
    """
    Evaluate a trained model using AUC and F1-score.
    """
    y_true = []
    y_pred = []

    for batch_x, batch_y in data_gen:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        if len(y_true) >= data_gen.samples:
            break

    auc = calculate_auc(y_true, y_pred)
    f1  = calculate_f1(y_true, y_pred)

    print(f"Model AUC      : {auc:.4f}")
    print(f"Model F1-Score : {f1:.4f}")
    return auc, f1


# ──────────────────────────────────────────────
# 4. MAIN — TRAINING PIPELINE
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── Config ────────────────────────────────
    DATA_DIR      = 'data/breast-histopathology-images'
    MODEL_SAVE    = 'models/resnet_model.h5'
    NUM_CLASSES   = 2
    BATCH_SIZE    = 32       # Good balance of speed and stability for ResNet50
    EPOCHS        = 10       # Sufficient for head-only fine-tuning
    MAX_IMAGES    = 20000    # ~7% of full dataset — fast iteration, meaningful signal
    IMAGE_PATH    = 'data/breast-histopathology-images/10253/sample.png'
    GRADCAM_LAYER = 'conv5_block3_out'  # Last conv layer in ResNet50
    CLASS_INDEX   = 1                   # Index for 'Cancer'
    CLASS_NAMES   = ['Normal', 'Cancer']

    # ── Build model ───────────────────────────
    print("\n[1/4] Building ResNet50 model (base layers frozen)...")
    model = build_resnet_model(NUM_CLASSES, freeze_base=True)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Callbacks ─────────────────────────────
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── Single training call ───────────────────
    print(f"\n[2/4] Starting training — {MAX_IMAGES} images, batch={BATCH_SIZE}, epochs={EPOCHS}...")
    from train_pipeline import train_model

    history = train_model(
        DATA_DIR,
        model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=MODEL_SAVE,
        max_images=MAX_IMAGES,
        callbacks=callbacks
    )

    print(f"\n[3/4] Training complete. Best model saved to: {MODEL_SAVE}")

    # ── Grad-CAM visualisation ─────────────────
    print("\n[4/4] Generating Grad-CAM overlay...")
    from explainable_ai import grad_cam

    overlay = grad_cam(IMAGE_PATH, MODEL_SAVE, GRADCAM_LAYER, CLASS_INDEX)
    cv2.imwrite('results/gradcam_overlay.png', overlay)
    print("Grad-CAM overlay saved to: results/gradcam_overlay.png")

    # ── Quick prediction demo ──────────────────
    predicted_class, confidence = predict_disease(IMAGE_PATH, MODEL_SAVE, CLASS_NAMES)