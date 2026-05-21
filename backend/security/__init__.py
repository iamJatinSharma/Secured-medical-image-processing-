"""Unified security interface for medical image encryption."""
from .chaos_encrypt import arnold_cat_map, arnold_cat_map_decrypt
from .logistic_encrypt import logistic_map_encrypt, logistic_map_decrypt
from .aes_encrypt import aes_encrypt, aes_decrypt

__all__ = [
    "arnold_cat_map",
    "arnold_cat_map_decrypt",
    "logistic_map_encrypt",
    "logistic_map_decrypt",
    "aes_encrypt",
    "aes_decrypt",
    "encrypt_image",
    "decrypt_image",
]


def encrypt_image(image, method="arnold", **kwargs):
    """Unified encryption interface.

    Args:
        image: numpy array (for arnold/logistic) or bytes (for aes).
        method: 'arnold', 'logistic', or 'aes'.
        **kwargs: method-specific parameters.
    """
    if method == "arnold":
        return arnold_cat_map(image, **kwargs)
    elif method == "logistic":
        return logistic_map_encrypt(image, **kwargs)
    elif method == "aes":
        img_bytes = image.tobytes()
        return aes_encrypt(img_bytes, kwargs.get("password", "default"))
    raise ValueError(f"Unknown method: {method}")


def decrypt_image(data, method="arnold", **kwargs):
    """Unified decryption interface.

    Args:
        data: encrypted image (numpy array for arnold/logistic, bytes for aes).
        method: 'arnold', 'logistic', or 'aes'.
        **kwargs: method-specific parameters.
    """
    if method == "arnold":
        return arnold_cat_map_decrypt(data, **kwargs)
    elif method == "logistic":
        return logistic_map_decrypt(data, **kwargs)
    elif method == "aes":
        return aes_decrypt(data, kwargs.get("password", "default"))
    raise ValueError(f"Unknown method: {method}")
