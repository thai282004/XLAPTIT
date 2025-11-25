import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gaussian_kernel_1d(sigma):
    """
    Gaussian 1D kernel.
    """
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 0.1

    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_separable(gray_img, sigma):
    """
    Gaussian blur tách trục (separable):
    - Blur ngang (horiz)
    - Blur dọc (vert)
    """
    img = gray_img.astype(np.float32)
    k = gaussian_kernel_1d(sigma)
    r = len(k) // 2

    h, w = img.shape

    # ---- Blur ngang ----
    padded_h = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    tmp = np.zeros_like(img, dtype=np.float32)
    for y in range(h):
        tmp[y, :] = np.convolve(padded_h[y, :], k, mode="valid")

    # ---- Blur dọc ----
    padded_v = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for x in range(w):
        out[:, x] = np.convolve(padded_v[:, x], k, mode="valid")

    return out


def pad_reflect(img, pad):
    """
    Reflect padding cho 2D hoặc 3D array.
    """
    if img.ndim == 2:
        return np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    elif img.ndim == 3:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")


def convolve2d(image, kernel):
    """
    Convolution 2D tối ưu:
    - Kernel nhỏ (<=25): dùng sliding_window_view (vector hóa)
    - Kernel lớn: fallback vòng for để tiết kiệm RAM
    """
    img = image.astype(np.float32)
    k = kernel.astype(np.float32)
    kh, kw = k.shape
    assert kh == kw and kh % 2 == 1

    pad = kh // 2

    # kernel lớn -> vòng for
    if kh > 25:
        padded = pad_reflect(img, pad)
        h, w = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        k_flipped = np.flip(k)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh, j:j+kw]
                out[i, j] = np.sum(region * k_flipped)
        return out

    # kernel nhỏ -> sliding_window_view
    padded = pad_reflect(img, pad)
    k_flipped = np.flip(k)
    windows = sliding_window_view(padded, (kh, kw))  # (H, W, kh, kw)
    out = np.sum(windows * k_flipped[None, None, :, :], axis=(2, 3))
    return out.astype(np.float32)


def gaussian_kernel(kernel_size=5, sigma=1.0):
    """Tạo kernel Gaussian 2D chuẩn hóa tổng = 1."""
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    k = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    s2 = 2 * sigma * sigma
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            val = math.exp(-(i * i + j * j) / s2)
            kernel[i + k, j + k] = val
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur(gray_img, kernel_size=5, sigma=1.0):
    """
    Gaussian blur “chuẩn” dùng separable convolution.
    """
    if sigma is None and kernel_size is not None:
        sigma = max((kernel_size - 1) / 6.0, 0.1)
    elif sigma is not None:
        sigma = float(sigma)
    else:
        sigma = 1.0

    return gaussian_blur_separable(gray_img, sigma)


def median_filter(gray_img, kernel_size=3):
    """
    Median filter tự cài.
    """
    assert kernel_size % 2 == 1
    img = gray_img.astype(np.float32)
    k = kernel_size // 2
    padded = pad_reflect(img, k)
    h, w = img.shape
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            out[i, j] = np.median(region)
    return out


def bilateral_filter(gray_img, sigma_s=2.0, sigma_r=25.0, window_size=None):
    """
    Bilateral filter phiên bản tối ưu:
    - Giới hạn window_size <= 11 để tránh quá nặng
    - Nếu ảnh lớn > ~1.2MP -> tự động fallback Gaussian blur
    """
    img = gray_img.astype(np.float32)
    h, w = img.shape

    # Ảnh quá lớn -> dùng Gaussian
    if h * w > 1_200_000:
        return gaussian_blur_separable(img, sigma_s)

    # window_size
    if window_size is None:
        window_size = int(6 * sigma_s + 1)
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, 11)

    r = window_size // 2
    padded = pad_reflect(img, r)

    # precompute Gaussian không gian
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    spatial = np.exp(-(xx * xx + yy * yy) / (2 * sigma_s * sigma_s))

    out = np.zeros_like(img, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            region = padded[y:y + window_size, x:x + window_size]
            center = padded[y + r, x + r]
            diff = region - center
            range_weight = np.exp(-(diff * diff) / (2 * sigma_r * sigma_r))
            weights = spatial * range_weight
            w_sum = np.sum(weights)
            if w_sum > 0:
                out[y, x] = np.sum(region * weights) / w_sum
            else:
                out[y, x] = center

    return out
