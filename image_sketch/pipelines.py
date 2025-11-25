import numpy as np

from .filters import (
    gaussian_blur,
    gaussian_blur_separable,
    bilateral_filter,
)
from .edges import sobel_edge, log_edge, canny_edge
from .io_utils import gray_to_rgb


def compute_histogram(gray_img):
    """
    Tính histogram cho ảnh xám [0,255] bằng NumPy.
    """
    hist, _ = np.histogram(gray_img.astype(np.uint8),
                           bins=256, range=(0, 256))
    return hist


def quantize_gray(gray_img, levels=4):
    """
    Lượng tử hóa mức xám để tạo vùng tô phẳng (cartoon-like).
    """
    gray = np.clip(gray_img.astype(np.float32), 0, 255)
    step = 256 / levels
    q = (gray // step) * step + step / 2.0
    return np.clip(q, 0, 255)


def painting_pipeline(gray_img,
                      method='canny',
                      smoothing_method='gaussian',
                      smoothing_sigma=1.0,
                      smoothing_params=None,
                      edge_params=None,
                      quant_levels=4):
    """
    CARTOON pipeline:
    1. Smoothing (Gaussian / Bilateral)
    2. Edge detection (Sobel / LoG / Canny)
    3. Invert edge → biên đen
    4. Quantize gray
    5. Overlay
    """
    if edge_params is None:
        edge_params = {}
    if smoothing_params is None:
        smoothing_params = {}

    # 1. smoothing
    if smoothing_method == 'gaussian':
        smoothed = gaussian_blur(gray_img, kernel_size=5, sigma=smoothing_sigma)
    elif smoothing_method == 'bilateral':
        sigma_s = smoothing_params.get('sigma_s', 2.0)
        sigma_r = smoothing_params.get('sigma_r', 25.0)
        smoothed = bilateral_filter(gray_img, sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        smoothed = gray_img

    # 2. edges
    if method == 'sobel':
        threshold = edge_params.get('threshold', 80)
        edges, _ = sobel_edge(smoothed, threshold=threshold)
    elif method == 'log':
        sigma = edge_params.get('sigma', 1.0)
        edges = log_edge(smoothed, sigma=sigma,
                         kernel_size=edge_params.get('kernel_size', 5),
                         threshold=edge_params.get('zero_cross_threshold', 0.0))
    elif method == 'canny':
        sigma = edge_params.get('sigma', 1.0)
        low = edge_params.get('low_ratio', 0.1)
        high = edge_params.get('high_ratio', 0.3)
        edges = canny_edge(gray_img, sigma=sigma,
                           low_ratio=low, high_ratio=high,
                           use_internal_smoothing=False,
                           pre_smoothed=smoothed)
    else:
        raise ValueError("Unknown method: " + method)

    e = edges / 255.0
    edge_mask = 1.0 - e   # biên đen

    qgray = quantize_gray(smoothed, levels=quant_levels)
    qnorm = qgray / 255.0

    result = qnorm * edge_mask * 255.0
    return result, edges


def pencil_sketch_bw(gray_img, sigma=10.0):
    """
    PENCIL SKETCH ĐEN TRẮNG.
    """
    gray = np.clip(gray_img.astype(np.float32), 0, 255)
    inv = 255.0 - gray

    sigma = float(sigma)
    if sigma <= 0:
        sigma = 0.1
    blurred = gaussian_blur_separable(inv, sigma)

    denom = 255.0 - blurred
    denom[denom < 1] = 1.0
    sketch = gray * 255.0 / denom
    sketch = np.clip(sketch, 0, 255)
    return sketch


def pencil_sketch_color(rgb_arr, sketch_gray):
    """
    PENCIL SKETCH MÀU.
    """
    s = np.clip(sketch_gray.astype(np.float32), 0, 255) / 255.0
    rgb = np.clip(rgb_arr.astype(np.float32), 0, 255) / 255.0
    out = rgb * s[..., None]
    return np.clip(out * 255.0, 0, 255)


def compute_histograms(original_gray, processed_gray):
    return compute_histogram(original_gray), compute_histogram(processed_gray)
