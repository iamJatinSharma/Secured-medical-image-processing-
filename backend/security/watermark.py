"""DCT-based invisible watermarking for medical images.

Implements frequency-domain watermarking using 8x8 block DCT on the
luminance (Y) channel of YCrCb color space. Uses only numpy for DCT
computation (no scipy/cv2 dependency for the core algorithm).
"""
import numpy as np
from typing import Optional


# Precompute 8x8 DCT-II matrix for fast block transforms
def _dct_matrix(n: int = 8) -> np.ndarray:
    """Compute the NxN DCT-II transformation matrix."""
    mat = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            if k == 0:
                mat[k, i] = 1.0 / np.sqrt(n)
            else:
                mat[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    return mat


_DCT8 = _dct_matrix(8)
_DCT8_T = _DCT8.T


def _dct2_block(block: np.ndarray) -> np.ndarray:
    """Apply 2D DCT to an 8x8 block: D * block * D^T."""
    return _DCT8 @ block @ _DCT8_T


def _idct2_block(block: np.ndarray) -> np.ndarray:
    """Apply 2D inverse DCT to an 8x8 block: D^T * block * D."""
    return _DCT8_T @ block @ _DCT8


def _rgb_to_ycrcb(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCrCb color space."""
    img = image.astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + (r - y) * 0.713
    cb = 128.0 + (b - y) * 0.564
    return np.stack([y, cr, cb], axis=-1)


def _ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    """Convert YCrCb back to RGB."""
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1] - 128.0
    cb = ycrcb[:, :, 2] - 128.0
    r = y + 1.403 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.773 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _text_to_bits(text: str) -> list:
    """Convert text string to list of bits."""
    bits = []
    for char in text:
        char_bits = format(ord(char), '08b')
        bits.extend([int(b) for b in char_bits])
    return bits


def _bits_to_text(bits: list) -> str:
    """Convert list of bits back to text."""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i + 8]
        char_code = int(''.join(str(b) for b in byte), 2)
        if char_code == 0:
            break
        chars.append(chr(char_code))
    return ''.join(chars)


def embed_watermark(image: np.ndarray, watermark_text: str, method: str = 'dct', alpha: float = 50.0) -> np.ndarray:
    """Embed invisible watermark into image using DCT.

    Args:
        image: Input image (H, W, 3) uint8
        watermark_text: Text to embed
        method: 'dct' (only supported method currently)
        alpha: Embedding strength (higher = more robust but more visible)

    Returns:
        Watermarked image (same shape, uint8)
    """
    if method != 'dct':
        raise ValueError(f"Unsupported method: {method}. Use 'dct'.")

    h, w = image.shape[:2]
    # Ensure dimensions are multiples of 8
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    # Convert to YCrCb
    ycrcb = _rgb_to_ycrcb(image[:h8, :w8])
    y_channel = ycrcb[:, :, 0].copy()

    # Convert text to bits
    bits = _text_to_bits(watermark_text)
    num_bits = len(bits)

    # Number of available 8x8 blocks
    blocks_h = h8 // 8
    blocks_w = w8 // 8
    total_blocks = blocks_h * blocks_w

    if num_bits > total_blocks:
        raise ValueError(
            f"Watermark too long ({num_bits} bits) for image "
            f"({total_blocks} available blocks). Max ~{total_blocks // 8} chars."
        )

    # Embed bits into DCT coefficients of 8x8 blocks
    # Use mid-frequency positions for robustness
    embed_pos = (4, 4)  # Mid-frequency DCT coefficient

    bit_idx = 0
    for bh in range(blocks_h):
        for bw in range(blocks_w):
            if bit_idx >= num_bits:
                break
            # Extract block
            r0, c0 = bh * 8, bw * 8
            block = y_channel[r0:r0 + 8, c0:c0 + 8].astype(np.float64)

            # Apply DCT
            dct_block = _dct2_block(block)

            # Embed bit by setting coefficient sign and magnitude
            bit = bits[bit_idx]
            if bit == 1:
                dct_block[embed_pos] = abs(alpha)
            else:
                dct_block[embed_pos] = -abs(alpha)

            # Inverse DCT
            modified_block = _idct2_block(dct_block)
            y_channel[r0:r0 + 8, c0:c0 + 8] = modified_block

            bit_idx += 1
        if bit_idx >= num_bits:
            break

    # Reconstruct image
    ycrcb[:, :, 0] = y_channel
    result = _ycrcb_to_rgb(ycrcb)

    # Place modified region back into full-size output
    output = image.copy()
    output[:h8, :w8] = result
    return output


def extract_watermark(image: np.ndarray, method: str = 'dct', length: int = 64) -> str:
    """Extract watermark from image.

    Args:
        image: Watermarked image (H, W, 3) uint8
        method: 'dct'
        length: Expected watermark length in characters (default 64)

    Returns:
        Extracted text string
    """
    if method != 'dct':
        raise ValueError(f"Unsupported method: {method}. Use 'dct'.")

    h, w = image.shape[:2]
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    # Convert to YCrCb
    ycrcb = _rgb_to_ycrcb(image[:h8, :w8])
    y_channel = ycrcb[:, :, 0]

    blocks_h = h8 // 8
    blocks_w = w8 // 8

    num_bits = length * 8
    embed_pos = (4, 4)

    bits = []
    bit_idx = 0
    for bh in range(blocks_h):
        for bw in range(blocks_w):
            if bit_idx >= num_bits:
                break
            r0, c0 = bh * 8, bw * 8
            block = y_channel[r0:r0 + 8, c0:c0 + 8].astype(np.float64)

            # Apply DCT
            dct_block = _dct2_block(block)

            # Read bit from coefficient sign
            coeff = dct_block[embed_pos]
            bits.append(1 if coeff >= 0 else 0)

            bit_idx += 1
        if bit_idx >= num_bits:
            break

    return _bits_to_text(bits)


def verify_watermark(image: np.ndarray, expected_text: str, method: str = 'dct') -> bool:
    """Verify if image contains expected watermark.

    Args:
        image: Watermarked image
        expected_text: Expected watermark text
        method: 'dct'

    Returns:
        True if extracted watermark matches expected text
    """
    extracted = extract_watermark(image, method=method, length=len(expected_text))
    return extracted.strip('\x00') == expected_text


def embed_visible_watermark(image: np.ndarray, text: str, position: str = 'bottom-right', opacity: float = 0.3) -> np.ndarray:
    """Add semi-transparent visible watermark text.

    Uses numpy-only rendering (no cv2 dependency). Draws text as a simple
    bitmap font overlay.

    Args:
        image: Input image (H, W, 3) uint8
        text: Text to overlay
        position: One of 'bottom-right', 'bottom-left', 'top-right', 'top-left'
        opacity: Blending opacity for the text overlay (0-1)

    Returns:
        Image with visible watermark
    """
    try:
        # Try using PIL for text rendering (available)
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(image)
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        h, w = image.shape[:2]
        font_size = max(12, min(h, w) // 20)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding = 10
        if position == 'bottom-right':
            x = w - text_w - padding
            y = h - text_h - padding
        elif position == 'bottom-left':
            x = padding
            y = h - text_h - padding
        elif position == 'top-right':
            x = w - text_w - padding
            y = padding
        else:  # top-left
            x = padding
            y = padding

        # Draw text with alpha
        alpha_val = int(opacity * 255)
        draw.text((x, y), text, fill=(255, 255, 255, alpha_val), font=font)

        # Composite
        pil_img = pil_img.convert('RGBA')
        result = Image.alpha_composite(pil_img, overlay)
        return np.array(result.convert('RGB'))

    except ImportError:
        # Fallback: simple numpy-based overlay (no PIL)
        result = image.copy()
        h, w = result.shape[:2]
        # Simple white rectangle as placeholder watermark indicator
        bar_h = max(20, h // 20)
        if 'bottom' in position:
            result[h - bar_h:h, :] = (
                (1 - opacity) * result[h - bar_h:h, :] + opacity * 255
            ).astype(np.uint8)
        else:
            result[0:bar_h, :] = (
                (1 - opacity) * result[0:bar_h, :] + opacity * 255
            ).astype(np.uint8)
        return result
