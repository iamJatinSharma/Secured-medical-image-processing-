"""
Module: image_security.py
Description: Provides functions for encrypting and decrypting medical images using symmetric key encryption (AES).
"""
import os

import cv2
import numpy as np
from cryptography.fernet import Fernet


# Generate and save a key (run once, keep safe)
def generate_key(key_path='secret.key'):
    key = Fernet.generate_key()
    with open(key_path, 'wb') as key_file:
        key_file.write(key)
    return key

# Load the key
def load_key(key_path='secret.key'):
    with open(key_path, 'rb') as key_file:
        key = key_file.read()
    return key

# Encrypt image
def encrypt_image(image_path, key_path='secret.key', output_path='encrypted.img'):
    key = load_key(key_path)
    f = Fernet(key)
    image = cv2.imread(image_path)
    image_bytes = image.tobytes()
    encrypted = f.encrypt(image_bytes)
    with open(output_path, 'wb') as file:
        file.write(encrypted)
    print(f"Image encrypted and saved to {output_path}")

# Decrypt image
def decrypt_image(encrypted_path, key_path='secret.key', shape=(256,256,3), output_path='decrypted.png'):
    key = load_key(key_path)
    f = Fernet(key)
    with open(encrypted_path, 'rb') as file:
        encrypted = file.read()
    decrypted_bytes = f.decrypt(encrypted)
    image = np.frombuffer(decrypted_bytes, dtype=np.uint8).reshape(shape)
    cv2.imwrite(output_path, image)
    print(f"Image decrypted and saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    # generate_key()  # Run once to generate key
    # encrypt_image('data/sample.png')
    # decrypt_image('encrypted.img', shape=(256,256,3))
    pass
