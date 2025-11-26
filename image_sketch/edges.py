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


def make_log_kernel(kernel_size=5, sigma=1.0):
    """Tạo kernel Laplacian-of-Gaussian (LoG) 2D.

    kernel_size: kích thước kernel (3, 5, 7, ...), phải lẻ.
    sigma: độ lệch chuẩn của Gaussian bên trong LoG.
    """
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 0.1

    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    radius = k // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)

    r2 = xx * xx + yy * yy
    sigma2 = sigma * sigma

    # LoG(x, y) ≈ (r^2 - 2σ^2) / σ^4 * exp(-r^2 / (2σ^2))
    norm = (r2 - 2.0 * sigma2) / (sigma2 * sigma2)
    kernel = norm * np.exp(-r2 / (2.0 * sigma2))

    # Chuẩn hóa về mean = 0 để không làm lệch sáng toàn ảnh
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def log_edge(gray_img, sigma=1.0, kernel_size=5, threshold=0.0):
    """Phát hiện biên bằng Laplacian-of-Gaussian (LoG) + zero-crossing.

    gray_img   : ảnh xám 2D.
    sigma      : độ lệch chuẩn Gaussian trong LoG.
    kernel_size: kích thước kernel LoG (lẻ, >=3).
    threshold  : ngưỡng chênh lệch (max - min) trong vùng lân cận
                 để zero-crossing được xem là biên thật.
    """
    img = gray_img.astype(np.float32)

    # 1) Tạo kernel LoG theo sigma & kernel_size
    k = make_log_kernel(kernel_size=kernel_size, sigma=sigma)

    # 2) Áp dụng LoG kernel lên ảnh
    log_resp = convolve2d(img, k)

    h, w = log_resp.shape
    edges = np.zeros_like(log_resp, dtype=np.float32)

    # 3) Dò zero-crossing trong lân cận 3×3
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = log_resp[i - 1:i + 2, j - 1:j + 2]
            pmax = patch.max()
            pmin = patch.min()
            if pmax > 0 and pmin < 0:
                if abs(pmax - pmin) > threshold:
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
