"""Arnold Cat Map chaos-based image encryption."""
import numpy as np


def arnold_cat_map(image: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Apply Arnold Cat Map to scramble image pixels.

    The Arnold Cat Map transforms pixel positions:
    [x'] = [1 1] [x] mod N
    [y']   [1 2] [y]

    Image must be square. If not square, pad to square, process, then crop.
    """
    h, w = image.shape[:2]
    N = max(h, w)

    if h != N or w != N:
        padded = np.zeros((N, N, *image.shape[2:]), dtype=image.dtype)
        padded[:h, :w] = image
    else:
        padded = image.copy()

    result = padded.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                new_x = (x + y) % N
                new_y = (x + 2 * y) % N
                temp[new_x, new_y] = result[x, y]
        result = temp

    return result[:h, :w]


def arnold_cat_map_decrypt(
    scrambled: np.ndarray,
    iterations: int = 20,
    original_shape: tuple = None,
) -> np.ndarray:
    """Decrypt Arnold Cat Map by applying inverse transformation.

    Inverse transform:
    [x'] = [ 2 -1] [x] mod N
    [y']   [-1  1] [y]
    """
    h, w = scrambled.shape[:2]
    N = max(h, w)

    if h != N or w != N:
        padded = np.zeros((N, N, *scrambled.shape[2:]), dtype=scrambled.dtype)
        padded[:h, :w] = scrambled
    else:
        padded = scrambled.copy()

    result = padded.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                new_x = (2 * x - y) % N
                new_y = (-x + y) % N
                temp[new_x, new_y] = result[x, y]
        result = temp

    if original_shape is not None:
        return result[: original_shape[0], : original_shape[1]]
    return result[:h, :w]
