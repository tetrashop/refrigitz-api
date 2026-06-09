import numpy as np
import math
from PIL import Image

def generate_obj(img):
    """
    تولید مش سه‌بعدی با رنگ رأسی (vertex color) و جابه‌جایی عمق
    - نگاشت استوانه‌ای (cylindrical) برای حفظ نسبت تصویر
    - هر رأس رنگ پیکسل متناظر خود را دارد
    - شعاع بر اساس روشنایی تغییر می‌کند
    """
    img = img.convert('RGB')
    width, height = img.size
    max_dim = 80  # دقت مش
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0

    # پارامترهای کره
    base_radius = 50.0
    depth_scale = 25.0

    # تولید رئوس با نگاشت استوانه‌ای:
    # y از 0 تا height-1 به زاویه قطبی theta (0 تا pi)
    # x از 0 تا width-1 به زاویه سمتی phi (0 تا 2pi)
    theta = np.linspace(0, math.pi, height, endpoint=True)
    phi = np.linspace(0, 2*math.pi, width, endpoint=False)
    phi, theta = np.meshgrid(phi, theta)

    # محاسبه روشنایی برای جابه‌جایی
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
    r = base_radius + gray * depth_scale

    # مختصات دکارتی
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)  # محور عمودی (قائم)
    z = r * np.sin(theta) * np.sin(phi)

    # رنگ‌ها (RGB) همان رنگ پیکسل
    colors = pixels.reshape(-1, 3)  # هر سطر یک رنگ

    # رئوس (با سه ستون)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # اندیس‌بندی برای مثلث‌ها (هر چهارضلعی → دو مثلث)
    faces = []
    for i in range(height-1):
        for j in range(width):
            # پیچیدن دور لبه‌ها
            j_next = (j+1) % width
            idx0 = i * width + j
            idx1 = i * width + j_next
            idx2 = (i+1) * width + j
            idx3 = (i+1) * width + j_next
            faces.append((idx0, idx1, idx3))
            faces.append((idx0, idx3, idx2))

    # ذخیره فایل OBJ با رنگ رأسی (فرمت: v x y z r g b)
    lines = ["# Refrigitz Olympic 3D Model with Vertex Colors"]
    for i, v in enumerate(vertices):
        c = colors[i]
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    return "\n".join(lines).encode('utf-8')
