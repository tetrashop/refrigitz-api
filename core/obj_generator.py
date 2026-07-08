import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=100):
    """
    مدل سه‌بعدی توخالی با پوستهٔ هوشمند (ضخامت متغیر) و لبه‌های منحنی.
    - سطح جلو: بر اساس روشنایی با هموارسازی قوی.
    - سطح پشت: جابجایی معکوس با ضخامت متناسب با ارتفاع (نواحی برجسته ضخیم‌تر).
    - دیواره‌های جانبی: انحنای سطح را در لبه‌ها دنبال می‌کنند.
    """
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width, height = img_gray.size
    
    # ۱. تغییر اندازه به grid_res (پردازش مستقیم)
    img_gray = img_gray.resize((grid_res, grid_res), Image.LANCZOS)
    img_rgb = img_rgb.resize((grid_res, grid_res), Image.LANCZOS)
    width = height = grid_res

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۲. محاسبهٔ عمق صاف و طبیعی (Unsharp Masking قوی)
    blurred = gaussian_filter(gray, sigma=5.0)          # پایهٔ بسیار صاف
    detail = gray - blurred                             # جزئیات ریز
    detail_smooth = gaussian_filter(detail, sigma=1.5)  # حذف نویز جزئیات
    depth = blurred + detail_smooth * 0.15              # فقط ۱۵٪ جزئیات برای شباهت
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    if invert:
        depth = 1.0 - depth

    # ۳. ساخت شبکهٔ منظم از رئوس (X, Y, Z)
    x = np.linspace(0, 100, width)
    y = np.linspace(0, 100, height)
    xx, yy = np.meshgrid(x, y)
    z_front = depth * height_scale

    # ضخامت هوشمند: متناسب با ارتفاع (حداقل ۵٪ ارتفاع برای جلوگیری از صفر شدن)
    max_thickness = height_scale * 0.12
    thickness = depth * max_thickness + 0.02 * height_scale
    z_back = z_front - thickness

    # ۴. ترکیب رئوس جلو و پشت
    vertices_front = np.stack([xx, yy, z_front], axis=-1).reshape(-1, 3)
    vertices_back  = np.stack([xx, yy, z_back], axis=-1).reshape(-1, 3)
    all_vertices = np.vstack([vertices_front, vertices_back])

    # ۵. رنگ‌ها (مستقیماً از تصویر اصلی)
    colors = np.array(img_rgb, dtype=np.float32) / 255.0
    colors_flat = colors.reshape(-1, 3)
    all_colors = np.vstack([colors_flat, colors_flat])

    # ۶. مثلث‌های جلو و پشت (شبکهٔ منظم)
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

    # ۷. دیواره‌های جانبی منحنی (با نمونه‌برداری از عمق)
    side_faces = []
    # دیوارهٔ بالا (y = 0) → ردیف اول
    for j in range(width - 1):
        a = j
        b = j + 1
        c = j + offset
        d = j + 1 + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ پایین (y = height-1) → ردیف آخر
    base = (height - 1) * width
    for j in range(width - 1):
        a = base + j
        b = base + j + 1
        c = base + j + offset
        d = base + j + 1 + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ چپ (x = 0) → ستون اول
    for i in range(height - 1):
        a = i * width
        b = (i + 1) * width
        c = i * width + offset
        d = (i + 1) * width + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    # دیوارهٔ راست (x = width-1) → ستون آخر
    for i in range(height - 1):
        a = i * width + (width - 1)
        b = (i + 1) * width + (width - 1)
        c = i * width + (width - 1) + offset
        d = (i + 1) * width + (width - 1) + offset
        side_faces.append([a, b, d])
        side_faces.append([a, d, c])
    side_faces = np.array(side_faces, dtype=int)

    all_faces = np.vstack([faces_front, faces_back, side_faces])

    # ۸. محاسبهٔ نرمال‌های رأسی (میانگین‌گیری از همهٔ وجه‌های متصل)
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

    # ۹. نوشتن فایل OBJ
    lines = ["# Refrigitz Olympic Hollow Shell with Smart Thickness"]
    for v, c, n in zip(all_vertices, all_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for tri in all_faces:
        lines.append(f"f {tri[0]+1}//{tri[0]+1} {tri[1]+1}//{tri[1]+1} {tri[2]+1}//{tri[2]+1}")

    return "\n".join(lines).encode('utf-8')
