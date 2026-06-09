import numpy as np, math
from PIL import Image

def process_image_2d_to_3d(img, fg=2):
    img = img.convert('RGB')
    width, height = img.size
    max_dim = 70
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)

    i_idx = np.arange(width, dtype=np.float32) - width/2
    j_idx = np.arange(height, dtype=np.float32) - height/2
    ii, jj = np.meshgrid(i_idx, j_idx)
    r_arr = np.sqrt(ii**2 + jj**2 + 1)
    theta = np.arccos(1.0 / r_arr)
    phi = np.arctan2(jj, ii)

    min_r, max_r = float(r_arr.min()), float(r_arr.max())
    min_t, max_t = float(theta.min()), float(theta.max())

    fg_val = 2
    cx = int((max_r - min_r) * fg_val + max_r + 1)
    cyp1 = int(round((max_t - min_t) * 2 + max_t + 1))
    c = np.zeros((cx, cyp1, 3), dtype=np.float32)

    lum = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
    denom = 1.0 + pixels[:,:,0] + pixels[:,:,1] + pixels[:,:,2] + 1e-9
    dr = max_r * ((ii + width/2 + 1) / (1 + np.sqrt(ii**2 + jj**2 + 1))) * 3.0 * 300.0 / denom
    dr = dr.astype(np.int32)

    theta_deg = theta * 180.0 / math.pi

    for bi in range(fg_val):
        for bj in range(fg_val):
            cxT = ((max_r - min_r) * bi + dr).astype(np.int32)
            cyT1 = np.round((max_t - min_t) * bj + theta_deg + 2.0).astype(np.int32)
            cyT2 = np.round((max_t - min_t) * bj + theta_deg - 2.0).astype(np.int32)

            mask1 = (cxT >= 0) & (cxT < cx) & (cyT1 >= 0) & (cyT1 < cyp1)
            mask2 = (cxT >= 0) & (cxT < cx) & (cyT2 >= 0) & (cyT2 < cyp1)

            if (bi + bj) % 2 == 0:
                c[cxT[mask1], cyT1[mask1]] = pixels[mask1]
            else:
                c[cxT[mask2], cyT2[mask2]] = pixels[mask2]

    out_w = int(width * fg_val)
    out_h = height
    e = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for bi in range(fg_val):
        for bj in range(fg_val):
            cx_idx = ((max_r - min_r) * bi + dr).astype(np.int32)
            cy_idx1 = np.round((max_t - min_t) * bj + theta_deg + 2.0).astype(np.int32)
            cy_idx2 = np.round((max_t - min_t) * bj + theta_deg - 2.0).astype(np.int32)
            mask_c1 = (cx_idx >= 0) & (cx_idx < cx) & (cy_idx1 >= 0) & (cy_idx1 < cyp1)
            mask_c2 = (cx_idx >= 0) & (cx_idx < cx) & (cy_idx2 >= 0) & (cy_idx2 < cyp1)

            if (bi + bj) % 2 == 0:
                for row in range(height):
                    mc = mask_c1[row]
                    col_indices = np.arange(width, dtype=np.int32)[mc]
                    e[row, bi * width + col_indices] = c[cx_idx[row, mc], cy_idx1[row, mc]]
            else:
                for row in range(height):
                    mc = mask_c2[row]
                    col_indices = np.arange(width, dtype=np.int32)[mc]
                    e[row, bi * width + col_indices] = c[cx_idx[row, mc], cy_idx2[row, mc]]

    e = np.clip(e, 0, 255).astype(np.uint8)
    return Image.fromarray(e, 'RGB')
