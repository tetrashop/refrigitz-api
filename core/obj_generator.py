import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=80):
    """
    مدل سه‌بعدی توخالی با سطح کاملاً صاف و رنگ‌آمیزی واقعی.
    - از شبکهٔ منظم (Uniform Grid) برای حذف مثلث‌های تیز استفاده می‌کند.
    - عمق = روشنایی با Unsharp Masking ملایم + هموارسازی قوی.
    - دیواره‌های جانبی حجم را می‌بندند.
    """
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width, height = img_gray.size
    
    # ۱. محدودیت ابعاد و تغییر اندازه به grid_res (پردازش مستقیم)
    max_dim = grid_res
    if width > max_dim or height > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
    img_gray = img_gray.resize((width, height), Image.LANCZOS)
    img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۲. محاسبهٔ عمق صاف و طبیعی
    blurred = gaussian_filter(gray, sigma=4.0)          # پایهٔ کاملاً صاف
    detail = gray - blurred                             # جزئیات ریز
    detail_smooth = gaussian_filter(detail, sigma=1.0)  # حذف نویز جزئیات
    depth = blurred + detail_smooth * 0.2               # فقط ۲۰٪ جزئیات برای شباهت
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    if invert:
        depth = 1.0 - depth

    # ۳. ساخت شبکهٔ منظم از رئوس
    x = np.linspace(0, 100, width)
    y = np.linspace(0, 100, height)
    xx, yy = np.meshgrid(x, y)
    zz = depth * height_scale

    # رئوس جلو و پشت
    vertices_front = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    shell_thickness = height_scale * 0.08
    vertices_back = np.stack([xx, yy, zz - shell_thickness], axis=-1).reshape(-1, 3)
    all_vertices = np.vstack([vertices_front, vertices_back])

    # رنگ‌ها (مستقیماً از تصویر اصلی)
    colors = np.array(img_rgb, dtype=np.float32) / 255.0
    colors_flat = colors.reshape(-1, 3)
    all_colors = np.vstack([colors_flat, colors_flat])

    # ۴. مثلث‌های جلو و پشت (شبکهٔ منظم)
    faces_front = []
    for i in range(height - 1):
        for j in range(width - 1):
            a = i * width + j
            b = a + 1
            c = (i + 1) * width + j
            d = c + 1
            faces_front.append([a, b, d])
            faces_front.append([a, d, c])
    faces_front = np.array(faces_front, dtype=int)

    offset = len(vertices_front)
    faces_back = faces_front[:, ::-1] + offset  # معکوس برای نرمال بیرون

    # ۵. دیواره‌های جانبی (نوارهای مثلثی)
    side_faces = []
    # دیوارهٔ بالا (y = 0)
    for j in range(width - 1):
        a = j
        b = j + 1
        c = j + offset
        d = j + 1 + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ پایین (y = height-1)
    base = (height - 1) * width
    for j in range(width - 1):
        a = base + j
        b = base + j + 1
        c = base + j + offset
        d = base + j + 1 + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ چپ (x = 0)
    for i in range(height - 1):
        a = i * width
        b = (i + 1) * width
        c = i * width + offset
        d = (i + 1) * width + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ راست (x = width-1)
    for i in range(height - 1):
        a = i * width + (width - 1)
        b = (i + 1) * width + (width - 1)
        c = i * width + (width - 1) + offset
        d = (i + 1) * width + (width - 1) + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    side_faces = np.array(side_faces, dtype=int)

    all_faces = np.vstack([faces_front, faces_back, side_faces])

    # ۶. محاسبهٔ نرمال‌های رأسی (میانگین‌گیری از همهٔ وجه‌های متصل)
    normals = np.zeros_like(all_vertices)
    for tri in all_faces:
        v0, v1, v2 = all_vertices[tri[0]], all_vertices[tri[1]], all_vertices[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in tri:
            normals[idx] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0, 0, 1])

    # ۷. نوشتن فایل OBJ
    lines = ["# Refrigitz Olympic Smooth Hollow Shell"]
    for v, c, n in zip(all_vertices, all_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for tri in all_faces:
        lines.append(f"f {tri[0]+1}//{tri[0]+1} {tri[1]+1}//{tri[1]+1} {tri[2]+1}//{tri[2]+1}")

    return "\n".join(lines).encode('utf-8')
