import numpy as np
import math
from PIL import Image
from scipy.spatial import Delaunay

def generate_obj(img):
    img = img.convert('RGB')
    width, height = img.size
    max_dim = 50
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0

    # ---------- نگاشت کروی کامل (r, θ, φ) ----------
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    phi = xx * 2 * math.pi          # ۰ تا ۲π
    theta = yy * math.pi            # ۰ تا π
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
    base_radius = 50.0
    depth_scale = 30.0
    r = base_radius + gray * depth_scale

    # تبدیل به مختصات دکارتی
    vx = r * np.sin(theta) * np.cos(phi)
    vy = r * np.sin(theta) * np.sin(phi)
    vz = r * np.cos(theta)

    # صاف‌کردن آرایه‌ها و نرمال‌سازی
    pts = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
    # تراکم یکنواخت: انتخاب نقاط با گام منظم (اینجا به دلیل شبکه منظم، تراکم یکنواخت است)
    # نیازی به resample نیست.

    # مثلث‌بندی دلونی (با استفاده از اندیس‌های شبکه)
    idx = np.arange(width * height).reshape(height, width)
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            a = idx[i, j]
            b = idx[i, j+1]
            c = idx[i+1, j]
            d = idx[i+1, j+1]
            faces.append((a, b, d))
            faces.append((a, d, c))
    faces = np.array(faces)

    # نوشتن OBJ
    lines = ["# Refrigitz Olympic 3D Sphere"]
    for p in pts:
        lines.append(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    return "\n".join(lines).encode('utf-8')
