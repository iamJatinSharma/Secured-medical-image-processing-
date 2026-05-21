"""AES-256-GCM authenticated encryption for image data."""
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from password using PBKDF2-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(password.encode("utf-8"))


def aes_encrypt(image_bytes: bytes, password: str) -> bytes:
    """Encrypt image bytes with AES-256-GCM.

    Returns: salt (16 bytes) + nonce (12 bytes) + ciphertext_with_tag
    """
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, image_bytes, None)
    return salt + nonce + ciphertext


def aes_decrypt(encrypted_data: bytes, password: str) -> bytes:
    """Decrypt AES-256-GCM encrypted data.

    Parses: salt (first 16 bytes) + nonce (next 12 bytes) + ciphertext_with_tag
    Raises cryptography.exceptions.InvalidTag on wrong password.
    """
    salt = encrypted_data[:16]
    nonce = encrypted_data[16:28]
    ciphertext = encrypted_data[28:]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
