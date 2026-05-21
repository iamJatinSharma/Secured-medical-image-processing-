"""
Module: train_pipeline.py
Description: Automated training pipeline for medical image models (data loading, augmentation, training, evaluation, saving).
"""

import os

import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from evaluation_metrics import calculate_auc, calculate_f1


def train_model(
    data_dir,
    model,
    epochs=10,
    batch_size=32,
    save_path='model.h5',
    max_images=None,        # Limit total training images for faster iteration
    callbacks=None          # Accept external callbacks (EarlyStopping, ReduceLROnPlateau, etc.)
):
    """
    Train a Keras model on image data organised in train/val subdirectories.

    Args:
        data_dir    : Root directory containing 'train/' and 'val/' subdirs.
        model       : Compiled Keras model.
        epochs      : Number of training epochs.
        batch_size  : Batch size for both train and val generators.
        save_path   : Path to save the best model checkpoint.
        max_images  : If set, limits training to this many images per epoch
                      via steps_per_epoch (does not delete or resample files).
        callbacks   : List of Keras callbacks. If None, defaults to ModelCheckpoint only.

    Returns:
        Keras History object.
    """

    # ── Augmentation for training only ────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # ── No augmentation for validation — only rescaling ───────────────────────
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # ── Steps per epoch (controls max_images) ─────────────────────────────────
    if max_images is not None:
        steps_per_epoch = min(max_images // batch_size, len(train_gen))
    else:
        steps_per_epoch = len(train_gen)   # Full dataset

    print(f"  → Training steps/epoch : {steps_per_epoch}  "
          f"(~{steps_per_epoch * batch_size} images)")
    print(f"  → Validation steps     : {len(val_gen)}")

    # ── Callbacks — merge default checkpoint with any external ones ───────────
    default_checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    if callbacks is not None:
        # Remove any duplicate ModelCheckpoint if caller already passed one
        external = [cb for cb in callbacks
                    if not isinstance(cb, ModelCheckpoint)]
        final_callbacks = [default_checkpoint] + external
    else:
        final_callbacks = [default_checkpoint]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=final_callbacks
    )

    return history