import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

def generate_obj(img, invert=False, grid_res=80):
    """
    مدل سه‌بعدی مبتنی بر لبه‌های هموار (جهت‌خنثی):
    - هر جا تغییرات شدت نور (لبه) وجود داشته باشد → ارتفاع ملایم
    - پس‌زمینه صاف و بدون برجستگی
    - روشن یا تاریک بودن ناحیه تأثیری در جهت ارتفاع ندارد
    """
    img = img.convert('L')  #灰度
    width, height = img.size
    max_dim = 100
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    gray = np.array(img, dtype=np.float32) / 255.0

    # ۱. محاسبهٔ قدرت لبه (Sobel)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)

    # ۲. هموارسازی قوی برای حذف نوک‌های تیز (تبدیل به تپه‌های ملایم)
    depth = gaussian_filter(edge_mag, sigma=4.0)

    # ۳. نرمال‌سازی به [0,1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    # ۴. معکوس‌سازی (اختیاری)
    if invert:
        depth = 1.0 - depth

    # ۵. افزایش وضوح به grid_res با درون‌یابی بیکوبیک
    from scipy.ndimage import zoom
    depth_map = zoom(depth, grid_res / max(width, height), order=2)[:grid_res, :grid_res]
    depth_map = np.clip(depth_map, 0, 1)

    # ۶. ساخت مش
    # تصویر رنگی اصلی را به اندازه grid_res تغییر دهیم
    img_rgb = img.resize((grid_res, grid_res), Image.LANCZOS)
    colors = np.array(img_rgb, dtype=np.float32) / 255.0

    x = np.linspace(0, 100, grid_res)
    y = np.linspace(0, 100, grid_res)
    xx, yy = np.meshgrid(x, y)
    zz = depth_map * 40.0  # مقیاس عمق

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # مثلث‌بندی
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

    # نرمال‌ها
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
    for v, c, n in zip(vertices, colors.reshape(-1, 3), normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')
