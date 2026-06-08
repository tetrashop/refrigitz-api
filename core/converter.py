import numpy as np
import math
from PIL import Image

def cart2sph(i, j, k=1):
    r = math.sqrt(i*i + j*j + k*k)
    if r == 0:
        return 0, 0, 0
    theta = math.acos(k / r)
    phi = math.atan2(j, i)
    return theta, phi, r

def process_image_2d_to_3d(img, fg=2):
    img = img.convert('RGB')
    width, height = img.size
    pixels = np.array(img, dtype=np.float32)

    # مرحله Initiate
    min_r, max_r = float('inf'), -float('inf')
    min_t, max_t = float('inf'), -float('inf')
    min_f, max_f = float('inf'), -float('inf')
    teta_arr = np.zeros((height, width))
    fi_arr = np.zeros((height, width))
    r_arr = np.zeros((height, width))
    
    for j in range(height):
        for i in range(width):
            teta, fi, r = cart2sph(i, j, 1)
            teta_deg = int(round(teta * 180 / math.pi))
            fi_deg = int(round(fi * 180 / math.pi))
            r_int = int(round(r))
            teta_arr[j, i] = teta_deg
            fi_arr[j, i] = fi_deg
            r_arr[j, i] = r_int
            min_r = min(min_r, r_int)
            max_r = max(max_r, r_int)
            min_t = min(min_t, teta_deg)
            max_t = max(max_t, teta_deg)
            min_f = min(min_f, fi_deg)
            max_f = max(max_f, fi_deg)

    cx = int((max_r - min_r) * fg + max_r + 1)
    cy = int(round((max_t - min_t) * fg + max_t + 1))
    c = np.zeros((cx, cy, 3), dtype=np.float32)

    # ContoObject (تسهیل‌شده)
    for j in range(height):
        for i in range(width):
            r = r_arr[j, i]
            t = teta_arr[j, i]
            f = fi_arr[j, i]
            dr = max_r * ((i + 1) / (1 + math.sqrt(i*i + j*j + 1))) * 3 * 300 / (1 + pixels[j, i, 0] + pixels[j, i, 1] + pixels[j, i, 2] + 1e-9)
            cx_pos = int((max_r - min_r) * 0 + dr)  # ساده‌سازی برای ii=0
            cy_pos = int(round((max_t - min_t) * 0 + t + 2))
            if 0 <= cx_pos < cx and 0 <= cy_pos < cy:
                c[cx_pos, cy_pos, :] = pixels[j, i, :]

    # ConvTo3D
    out_width = int(width * fg)
    out_height = height
    e = np.zeros((out_height, out_width, 3), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            dr = max_r * ((i + 1) / (1 + math.sqrt(i*i + j*j + 1))) * 3 * 300 / (1 + pixels[j, i, 0] + pixels[j, i, 1] + pixels[j, i, 2] + 1e-9)
            cx_pos = int((max_r - min_r) * 0 + dr)
            cy_pos = int(round((max_t - min_t) * 0 + teta_arr[j, i] + 2))
            if 0 <= cx_pos < cx and 0 <= cy_pos < cy:
                if (0 + 0) % 2 == 0:
                    if 0 <= i + int(0 * width) < out_width and 0 <= j < out_height:
                        e[j, i + int(0 * width)] = c[cx_pos, cy_pos, :]

    # نرمال‌سازی نهایی
    e = np.clip(e, 0, 255)
    e = e.astype(np.uint8)
    result = Image.fromarray(e)
    return result
