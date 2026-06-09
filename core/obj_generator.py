import numpy as np
import math
from io import BytesIO
from PIL import Image

def cart2sph(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0, 0, 0
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return theta, phi, r

def generate_obj(img):
    """
    تولید فایل OBJ با مش کروی:
    - هر پیکسل یک نقطه در فضای سه‌بعدی می‌شود (بر اساس مختصات کروی)
    - سپس با مثلث‌بندی نقاط همسایه، یک سطح حجمی می‌سازیم
    """
    img = img.convert('RGB')
    width, height = img.size
    max_dim = 60  # محدودیت برای OBJ
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height))
    
    pixels = np.array(img, dtype=np.float32)
    vertices = []
    colors = []
    
    # ۱. تبدیل هر پیکسل به مختصات کروی و سپس به دکارتی سه‌بعدی
    for y in range(height):
        for x in range(width):
            # مختصات دکارتی با عمق بر اساس روشنایی
            gray = 0.299 * pixels[y, x, 0] + 0.587 * pixels[y, x, 1] + 0.114 * pixels[y, x, 2]
            depth = gray / 255.0 * 50.0  # مقیاس عمق
            # مختصات کروی با r=100+عمق
            r = 100.0 + depth
            theta = (y / height) * math.pi  # قائم (0 تا pi)
            phi = (x / width) * 2 * math.pi  # افقی (0 تا 2pi)
            # تبدیل به دکارتی
            vx = r * math.sin(theta) * math.cos(phi)
            vy = r * math.sin(theta) * math.sin(phi)
            vz = r * math.cos(theta)
            vertices.append((vx, vy, vz))
            colors.append((pixels[y, x, 0]/255, pixels[y, x, 1]/255, pixels[y, x, 2]/255))
    
    # ۲. تولید وجه‌ها (دو مثلث برای هر چهار سلول مجاور)
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            i0 = y * width + x + 1
            i1 = y * width + (x + 1) + 1
            i2 = (y + 1) * width + x + 1
            i3 = (y + 1) * width + (x + 1) + 1
            faces.append((i0, i1, i3))
            faces.append((i0, i3, i2))
    
    # ساخت OBJ
    lines = ["# Refrigitz Olympic 3D Model"]
    for v in vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for c in colors:
        lines.append(f"vt {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")  # مختصات بافت (اختیاری)
    for f in faces:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    
    return "\n".join(lines).encode('utf-8')
