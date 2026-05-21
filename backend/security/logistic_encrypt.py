"""Logistic Map chaos-based image encryption via XOR."""
import numpy as np


def _generate_chaotic_sequence(length: int, key: float = 3.99, x0: float = 0.5) -> np.ndarray:
    """Generate a chaotic sequence using the logistic map.

    x_{n+1} = key * x_n * (1 - x_n)

    Discards first 100 transient values to reach chaotic regime.
    Returns integer values in [0, 255].
    """
    transient = 100
    total = transient + length
    x = x0
    sequence = np.empty(total, dtype=np.float64)
    for i in range(total):
        x = key * x * (1.0 - x)
        sequence[i] = x
    # Discard transients, convert to byte range
    chaotic = sequence[transient:]
    return (chaotic * 255).astype(np.uint64) & 0xFF


def logistic_map_encrypt(
    image: np.ndarray, key: float = 3.99, x0: float = 0.5
) -> np.ndarray:
    """Encrypt image using logistic map XOR.

    Each pixel is XORed with a value from the chaotic sequence.
    """
    flat = image.flatten().astype(np.uint8)
    seq = _generate_chaotic_sequence(len(flat), key=key, x0=x0).astype(np.uint8)
    encrypted = np.bitwise_xor(flat, seq)
    return encrypted.reshape(image.shape).astype(image.dtype)


def logistic_map_decrypt(
    encrypted: np.ndarray, key: float = 3.99, x0: float = 0.5
) -> np.ndarray:
    """Decrypt image encrypted with logistic map XOR.

    XOR is symmetric: applying the same operation with the same key decrypts.
    """
    return logistic_map_encrypt(encrypted, key=key, x0=x0)
