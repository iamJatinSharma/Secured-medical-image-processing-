"""
Module: image_segmentation.py
Description: Functions for segmenting medical images (thresholding, U-Net deep learning segmentation).
"""
import cv2
import numpy as np
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, UpSampling2D,
                                     concatenate)
from tensorflow.keras.models import Model


def threshold_segmentation(image_path, output_path, thresh_val=128):
    image = cv2.imread(image_path, 0)
    _, segmented = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, segmented)
    print(f"Image segmented and saved to {output_path}")

def build_unet(input_shape=(224, 224, 3)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_segment(image_path, model_path, output_path):
    from tensorflow.keras.models import load_model
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    model = load_model(model_path)
    mask = model.predict(image)[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (224, 224))
    cv2.imwrite(output_path, mask)
    print(f"U-Net segmentation mask saved to {output_path}")

if __name__ == "__main__":
    # threshold_segmentation('data/sample.png', 'results/segmented.png')
    # unet_segment('data/sample.png', 'models/unet_model.h5', 'results/unet_mask.png')
    pass
