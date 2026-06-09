import numpy as np
import math
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import sobel, distance_transform_edt

def process_image_2d_to_3d(img, fg=2):
    """تبدیل تصویر به تصویر سه‌بعدی با درون‌یابی منحنی‌وار"""
    img = img.convert('L')  #灰度
    width, height = img.size
    # محدودیت برای اجرا
    max_dim = 80
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64) / 255.0

    # مختصات کروی اولیه
    xx, yy = np.meshgrid(np.arange(width) - width/2, np.arange(height) - height/2)
    r = np.sqrt(xx**2 + yy**2 + 1.0)
    theta = np.arccos(1.0 / r)
    phi = np.arctan2(yy, xx)

    # نگاشت به تصویر خروجی
    out_w = int(width * fg)
    out_h = height
    map_x = (phi / (2*math.pi) + 0.5) * (out_w - 1)
    map_y = (theta / math.pi) * (out_h - 1)

    # تصویر خروجی اولیه (با حفره)
    result = np.zeros((out_h, out_w), dtype=np.float64)
    weights = np.zeros((out_h, out_w), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            nx = int(round(map_x[y, x]))
            ny = int(round(map_y[y, x]))
            if 0 <= nx < out_w and 0 <= ny < out_h:
                result[ny, nx] += pixels[y, x]
                weights[ny, nx] += 1
    mask = weights > 0
    result[mask] /= weights[mask]

    # ========== استراتژی درون‌یابی ==========
    # 1. تشخیص منحنی‌های لبه با گرادیان
    edges = sobel(pixels, axis=0)**2 + sobel(pixels, axis=1)**2
    edges = edges > np.percentile(edges, 80)

    # 2. برای هر پیکسل خالی، نزدیک‌ترین نقطه غیرخالی را در راستای گرادیان بیاب
    empty_mask = (weights == 0)
    if np.any(empty_mask):
        # فاصله تا نزدیکترین نقطه پر
        dist, idx = distance_transform_edt(empty_mask, return_indices=True)
        # idx[0], idx[1] مختصات نزدیکترین نقطه پر برای هر پیکسل خالی
        for i in range(out_h):
            for j in range(out_w):
                if empty_mask[i, j]:
                    ni, nj = idx[0, i, j], idx[1, i, j]
                    # درون‌یابی اسپلاین در همسایگی ۳×۳ (ساده‌شده: استفاده از مقدار نزدیکترین)
                    result[i, j] = result[ni, nj]
    # 3. هموارسازی و نرمال‌سازی
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, 'L')
