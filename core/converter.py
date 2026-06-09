import numpy as np
import math
from PIL import Image

def cart2sph(x, y, z):
    """تبدیل دکارتی به کروی (r, theta, phi)"""
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0, 0, 0
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return theta, phi, r

def process_image_2d_to_3d(img, fg=2):
    """
    تبدیل تصویر دوبعدی به سه‌بعدی با الگوریتم اصلی Refrigitz:
    1. محاسبه مختصات کروی هر پیکسل
    2. پخش آن‌ها در تصویر بزرگ‌تر (عرض = width * fg)
    """
    img = img.convert('RGB')
    width, height = img.size
    # محدود کردن ابعاد برای اجرا در ۱۰ ثانیه
    max_dim = 80
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height))
    
    pixels = np.array(img, dtype=np.float32)
    
    # محاسبه مینیمم و ماکسیمم r و theta
    min_r, max_r = float('inf'), -float('inf')
    min_t, max_t = float('inf'), -float('inf')
    r_arr = np.zeros((height, width))
    t_arr = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            theta, phi, r = cart2sph(x - width/2, y - height/2, 1.0)
            r_arr[y, x] = r
            t_arr[y, x] = theta
            if r < min_r: min_r = r
            if r > max_r: max_r = r
            if theta < min_t: min_t = theta
            if theta > max_t: max_t = theta
    
    if max_r == min_r: max_r += 1
    if max_t == min_t: max_t += 1
    
    # ابعاد تصویر خروجی
    out_w = int(width * fg)
    out_h = height
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=int)
    
    # پخش نقاط در خروجی (با وزن‌دهی ساده)
    for y in range(height):
        for x in range(width):
            r_norm = (r_arr[y, x] - min_r) / (max_r - min_r)
            t_norm = (t_arr[y, x] - min_t) / (max_t - min_t)
            # نگاشت به مختصات خروجی (افقی بر اساس phi، عمودی بر اساس theta)
            new_x = int(r_norm * (out_w - 1))
            new_y = int(t_norm * (out_h - 1))
            if 0 <= new_x < out_w and 0 <= new_y < out_h:
                result[new_y, new_x] += pixels[y, x]
                count[new_y, new_x] += 1
    
    # میانگین‌گیری نقاط روی هم افتاده
    mask = count > 0
    for c in range(3):
        result[:, :, c][mask] /= count[mask]
    
    # پر کردن نقاط خالی با درونیابی
    from scipy import ndimage
    if np.any(count == 0):
        for c in range(3):
            result[:, :, c] = ndimage.zoom(
                ndimage.zoom(result[:, :, c], 0.5, order=1),
                2.0, order=1
            )[:out_h, :out_w]
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, 'RGB')
