import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import sobel

def generate_obj(img):
    """
    تولید مش سه‌بعدی با نقش برجستهٔ طبیعی:
    - عمق = ترکیب روشنایی و شدت لبه‌ها
    - حفظ نسبت ابعاد
    - رنگ رأسی
    """
    img = img.convert('RGB')
    width, height = img.size
    max_res = 100
    if width > max_res or height > max_res:
        ratio = min(max_res / width, max_res / height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    pixels = np.array(img, dtype=np.float32) / 255.0

    # روشنایی
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

    # قدرت لبه (Sobel)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-9)  # نرمال‌سازی به [0,1]

    # ترکیب عمق: ۷۰٪ روشنایی + ۳۰٪ لبه
    depth = 0.7 * gray + 0.3 * edge_mag

    # افزودن بافت ریز (اختیاری) برای جلوگیری از همواری
    np.random.seed(42)
    micro_noise = np.random.normal(0, 0.005, size=depth.shape)
    depth = np.clip(depth + micro_noise, 0, 1)

    # مقیاس‌بندی
    max_dim = max(width, height)
    scale_x = 100.0 / max_dim
    scale_y = 100.0 / max_dim
    scale_z = 40.0

    x = np.arange(width) * scale_x
    y = np.arange(height) * scale_y
    xx, yy = np.meshgrid(x, y)
    zz = depth * scale_z

    # رئوس و رنگ‌ها
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    colors = pixels.reshape(-1, 3)

    # مثلث‌بندی
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            idx0 = i * width + j
            idx1 = i * width + (j + 1)
            idx2 = (i + 1) * width + j
            idx3 = (i + 1) * width + (j + 1)
            faces.append((idx0, idx1, idx3))
            faces.append((idx0, idx3, idx2))

    # OBJ
    lines = ["# Refrigitz Olympic 3D Relief with Edge Enhancement"]
    for v, c in zip(vertices, colors):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    return "\n".join(lines).encode('utf-8')
