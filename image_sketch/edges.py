import numpy as np

from .filters import (
    convolve2d,
    gaussian_blur,
    gaussian_blur_separable,
)


def sobel_kernels():
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)
    return gx, gy


def sobel_edge(gray_img, threshold=50):
    """
    Phát hiện biên bằng Sobel.
    """
    gx_k, gy_k = sobel_kernels()
    gx = convolve2d(gray_img, gx_k)
    gy = convolve2d(gray_img, gy_k)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_norm = mag / (mag.max() + 1e-6) * 255.0
    edges = (mag_norm >= threshold).astype(np.float32) * 255.0
    return edges, mag_norm


def log_edge(gray_img, sigma=1.0, kernel_size=5, threshold=0.0):
    """
    LoG edge detection - BẢN TỐI ƯU:
    - Gaussian blur (separable) với sigma
    - Laplacian 3x3
    - Zero-crossing
    """
    # 1. Làm mịn bằng Gaussian separable
    blurred = gaussian_blur_separable(gray_img, sigma)

    # 2. Laplacian kernel 3x3
    lap = np.array([[0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0]], dtype=np.float32)

    log_resp = convolve2d(blurred, lap)

    # 3. Zero-crossing
    h, w = log_resp.shape
    edges = np.zeros_like(log_resp, dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = log_resp[i - 1:i + 2, j - 1:j + 2]
            if patch.max() > 0 and patch.min() < 0:
                if abs(patch.max() - patch.min()) > threshold:
                    edges[i, j] = 255.0

    return edges


def gradient_sobel(gray_img):
    """Tính gradient magnitude & direction bằng Sobel."""
    gx_k, gy_k = sobel_kernels()
    gx = convolve2d(gray_img, gx_k)
    gy = convolve2d(gray_img, gy_k)
    mag = np.sqrt(gx * gx + gy * gy)
    direction = np.arctan2(gy, gx)  # radian
    return mag, direction


def non_maximum_suppression(mag, direction):
    """
    NMS cho Canny.
    """
    h, w = mag.shape
    out = np.zeros((h, w), dtype=np.float32)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255

            # 0 độ
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            # 45 độ
            elif 22.5 <= angle[i, j] < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            # 90 độ
            elif 67.5 <= angle[i, j] < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            # 135 độ
            elif 112.5 <= angle[i, j] < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                out[i, j] = mag[i, j]
            else:
                out[i, j] = 0.0

    return out


def double_threshold_and_hysteresis(img, low, high):
    """
    Double threshold + hysteresis cho Canny.
    """
    strong = 255.0
    weak = 50.0

    res = np.zeros_like(img, dtype=np.float32)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if res[i, j] == weak:
                neighborhood = res[i - 1:i + 2, j - 1:j + 2]
                if (neighborhood == strong).any():
                    res[i, j] = strong
                else:
                    res[i, j] = 0.0
    res[res != strong] = 0.0
    return res


def canny_edge(gray_img, sigma=1.0, low_ratio=0.1, high_ratio=0.3,
               use_internal_smoothing=True, pre_smoothed=None):
    """
    Full Canny pipeline.
    """
    # 1. smoothing
    if use_internal_smoothing:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        smoothed = gaussian_blur(gray_img, ksize, sigma)
    else:
        smoothed = pre_smoothed

    # 2. Gradient
    mag, direction = gradient_sobel(smoothed)
    mag = mag / (mag.max() + 1e-6) * 255.0

    # 3. NMS
    nms = non_maximum_suppression(mag, direction)

    # 4. Double threshold + hysteresis
    high = high_ratio * 255.0
    low = low_ratio * 255.0
    edges = double_threshold_and_hysteresis(nms, low, high)
    return edges
