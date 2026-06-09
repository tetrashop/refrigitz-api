import numpy as np
import math
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

def generate_obj(img):
    """تولید مش OBJ با درون‌یابی، تراکم یکسان و نرمال‌سازی"""
    img = img.convert('RGB')
    width, height = img.size
    max_dim = 50  # برای OBJ
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64) / 255.0

    # ---- مرحله ۱: ابر نقاط اولیه با عمق ----
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    gray = 0.299*pixels[:,:,0] + 0.587*pixels[:,:,1] + 0.114*pixels[:,:,2]
    depth = gray * 30.0  # مقیاس عمق

    points = np.stack([xx, yy, depth], axis=-1).reshape(-1, 3)

    # ---- مرحله ۲: تشخیص منحنی‌های لبه و برازش اسپلاین ----
    # (ساده‌شده: استفاده از گرادیان برای وزن‌دهی)
    edges = np.gradient(gray)
    edge_mag = np.sqrt(edges[0]**2 + edges[1]**2)
    edge_mask = edge_mag > np.percentile(edge_mag, 75)

    # ---- مرحله ۳: یکسان‌سازی تراکم با شعاع فیلتر ----
    # محاسبه فاصلهٔ متوسط بین نقاط
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(points)
    dist, _ = neigh.kneighbors(points)
    avg_dist = np.mean(dist[:, 1])

    # نمونه‌برداری مجدد روی شبکه با گام avg_dist
    grid_x = np.arange(0, width, avg_dist)
    grid_y = np.arange(0, height, avg_dist)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    # درون‌یابی عمق
    grid_depth = griddata((xx.ravel(), yy.ravel()), depth.ravel(),
                          (grid_xx, grid_yy), method='linear', fill_value=0.0)

    # ---- مرحله ۴: نرمال‌سازی ----
    # مقیاس به [0,1] در هر محور
    pts = []
    colors = []
    for i in range(len(grid_y)):
        for j in range(len(grid_x)):
            z = grid_depth[i, j]
            if z > 0:  # نقاط معتبر
                pts.append([grid_xx[i,j], grid_yy[i,j], z])
                # رنگ را از تصویر اصلی درونیابی کن
                col_x = min(int(grid_xx[i,j]), width-1)
                col_y = min(int(grid_yy[i,j]), height-1)
                colors.append(pixels[col_y, col_x])

    pts = np.array(pts)
    if len(pts) < 4:
        # fallback ساده
        return b"# No points"

    # نرمال‌سازی
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts *= 100  # مقیاس نهایی

    # ---- مرحله ۵: مثلث‌بندی دلونی و صادرات OBJ ----
    tri = Delaunay(pts[:, :2])  # مثلث‌بندی در XY
    faces = tri.simplices

    lines = ["# Refrigitz Olympic 3D - Interpolated"]
    for p in pts:
        lines.append(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    return "\n".join(lines).encode('utf-8')
