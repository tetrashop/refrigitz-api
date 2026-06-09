import numpy as np
import math
from PIL import Image
from scipy.ndimage import map_coordinates

def process_image_2d_to_3d(img, fg=2):
    img = img.convert('RGB')
    width, height = img.size
    # محدودیت ابعاد برای اجرا در ۱۰ ثانیه
    max_dim = 80
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)

    # ---------- Initiate (محاسبه r, θ, φ) ----------
    i_idx = np.arange(width) - width/2
    j_idx = np.arange(height) - height/2
    ii, jj = np.meshgrid(i_idx, j_idx)
    r_arr = np.sqrt(ii**2 + jj**2 + 1)
    # θ ∈ [0, π], φ ∈ (-π, π]
    theta = np.arccos(1.0 / r_arr)
    phi = np.arctan2(jj, ii)

    min_r, max_r = r_arr.min(), r_arr.max()
    min_t, max_t = theta.min(), theta.max()
    min_f, max_f = phi.min(), phi.max()

    # ---------- ContoObject (پرشدن c) ----------
    fg = 2  # ثابت در الگوریتم اصلی
    cx = int((max_r - min_r) * fg + max_r + 1)
    cyp1 = int(round((max_t - min_t) * 2 + max_t + 1))  # توجه: 2*delta + max
    c = np.zeros((cx, cyp1, 3), dtype=np.float32)

    # dr: جابجایی شعاعی بر اساس روشنایی (از کد اصلی)
    luminance = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
    dr = max_r * ((ii + width/2 + 1) / (1 + np.sqrt(ii**2 + jj**2 + 1))) * 3.0 * 300.0 / (1.0 + pixels[:,:,0] + pixels[:,:,1] + pixels[:,:,2] + 1e-9)
    dr = dr.astype(np.int32)

    for ii_block in range(fg):
        for jj_block in range(fg):
            cxT = (max_r - min_r) * ii_block + dr
            cyT1 = np.round((max_t - min_t) * jj_block + theta * 180 / math.pi + 2.0).astype(np.int32)
            cyT2 = np.round((max_t - min_t) * jj_block + theta * 180 / math.pi - 2.0).astype(np.int32)

            mask = (cxT >= 0) & (cxT < cx) & (cyT1 >= 0) & (cyT1 < cyp1) & (cyT2 >= 0) & (cyT2 < cyp1)
            if (ii_block + jj_block) % 2 == 0:
                c[cxT[mask], cyT1[mask], :] = pixels[mask]
            else:
                c[cxT[mask], cyT2[mask], :] = pixels[mask]

    # ---------- ConvTo3D (بازسازی تصویر خروجی) ----------
    out_w = int(width * fg)
    out_h = height
    e = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for ii_block in range(fg):
        for jj_block in range(fg):
            cx_idx = (max_r - min_r) * ii_block + dr
            cy_idx1 = np.round((max_t - min_t) * jj_block + theta * 180 / math.pi + 2.0).astype(np.int32)
            cy_idx2 = np.round((max_t - min_t) * jj_block + theta * 180 / math.pi - 2.0).astype(np.int32)

            mask_c = (cx_idx >= 0) & (cx_idx < cx) & (cy_idx1 >= 0) & (cy_idx1 < cyp1)
            if (ii_block + jj_block) % 2 == 0:
                for row in range(height):
                    col_mask = mask_c[row, :]
                    e[row, ii_block * width + np.arange(width)[col_mask], :] = c[cx_idx[row, col_mask], cy_idx1[row, col_mask], :]
            else:
                for row in range(height):
                    col_mask = mask_c[row, :]
                    e[row, ii_block * width + np.arange(width)[col_mask], :] = c[cx_idx[row, col_mask], cy_idx2[row, col_mask], :]

    e = np.clip(e, 0, 255).astype(np.uint8)
    return Image.fromarray(e, 'RGB')
