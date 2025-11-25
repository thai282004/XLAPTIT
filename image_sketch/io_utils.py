import numpy as np
from PIL import Image


def load_image(path):
    """Load image as PIL Image and numpy float32 array (RGB, 0-255)."""
    pil_img = Image.open(path).convert("RGB")
    arr = np.array(pil_img).astype(np.float32)
    return pil_img, arr


def save_image(path, arr):
    """Save a numpy array (0-255) as image."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def rgb_to_gray(rgb_arr):
    """
    Manual grayscale conversion.
    I = 0.299 R + 0.587 G + 0.114 B
    """
    r = rgb_arr[..., 0]
    g = rgb_arr[..., 1]
    b = rgb_arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def gray_to_rgb(gray_arr):
    """Stack gray 2D array into 3-channel RGB array."""
    gray = np.clip(gray_arr, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
