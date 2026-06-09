import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, zoom

def generate_obj(img_pil, invert=False, grid_res=80):
    """
    مدل سه‌بعدی مبتنی بر لبه‌های هموار (جهت‌خنثی):
    - از یک تصویر رنگی برای رنگ‌ها و از نگاشت خاکستری آن برای عمق استفاده می‌شود.
    - هر جا تغییرات شدت نور (لبه) وجود داشته باشد → ارتفاع ملایم
    - پس‌زمینه صاف و بدون برجستگی
    """
    # ذخیرهٔ تصویر رنگی اصلی برای رنگ‌ها
    img_rgb = img_pil.convert('RGB')
    # تصویر خاکستری برای عمق
    img_gray = img_rgb.convert('L')

    width, height = img_gray.size
    max_dim = 100
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img_gray = img_gray.resize((width, height), Image.LANCZOS)
        img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۱. قدرت لبه (Sobel)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)

    # ۲. هموارسازی قوی
    depth = gaussian_filter(edge_mag, sigma=4.0)

    # ۳. نرمال‌سازی
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    if invert:
        depth = 1.0 - depth

    # ۴. تنظیم اندازه به grid_res × grid_res (درون‌یابی)
    depth_map = zoom(depth, (grid_res / height, grid_res / width), order=2)[:grid_res, :grid_res]
    depth_map = np.clip(depth_map, 0, 1)

    # ۵. تصویر رنگی به اندازهٔ شبکه
    img_rgb_resized = img_rgb.resize((grid_res, grid_res), Image.LANCZOS)
    colors = np.array(img_rgb_resized, dtype=np.float32) / 255.0  # (grid_res, grid_res, 3)

    # ۶. ساخت مش
    x = np.linspace(0, 100, grid_res)
    y = np.linspace(0, 100, grid_res)
    xx, yy = np.meshgrid(x, y)
    zz = depth_map * 40.0  # مقیاس عمق

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (N, 3)
    face_colors = colors.reshape(-1, 3)  # (N, 3)

    faces = []
    for i in range(grid_res - 1):
        for j in range(grid_res - 1):
            a = i * grid_res + j
            b = a + 1
            c = (i+1) * grid_res + j
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    faces = np.array(faces)

    # نرمال‌های رأسی
    normals = np.zeros_like(vertices)
    cnt = np.zeros(len(vertices), dtype=int)
    for tri in faces:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in tri:
            normals[idx] += n
            cnt[idx] += 1
    mask = cnt > 0
    normals[mask] /= cnt[mask, np.newaxis]
    normals[~mask] = np.array([0, 0, 1])

    # نوشتن OBJ
    lines = ["# Refrigitz Edge-Based Relief (Direction Neutral)"]
    for v, c, n in zip(vertices, face_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')
