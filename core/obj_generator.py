import numpy as np
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter, zoom
from scipy.spatial import KDTree

def generate_obj(img, upsample=2):
    """
    تولید مش سه‌بعدی با سطح هموار و طبیعی:
    - عمق = ترکیب روشنایی + لبه (edge enhanced)
    - افزایش رزولوشن مش با درون‌یابی (upsample)
    - محاسبه نرمال‌های رأسی برای سایه‌زنی صاف
    """
    img = img.convert('RGB')
    width, height = img.size
    max_res = 80  # وضوح پایه
    if width > max_res or height > max_res:
        ratio = min(max_res / width, max_res / height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    pixels = np.array(img, dtype=np.float32) / 255.0

    # روشنایی
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

    # تشخیص لبه (Sobel)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_mag /= (edge_mag.max() + 1e-9)

    # ترکیب عمق
    depth = 0.7 * gray + 0.3 * edge_mag

    # فیلتر دوطرفه ساده (حفظ لبه، کاهش نویز)
    # با دو بار هموارسازی گوسی ملایم و ترکیب با تصویر اصلی
    depth_smooth = gaussian_filter(depth, sigma=0.8)
    depth = depth * 0.6 + depth_smooth * 0.4  # نگه‌داشتن لبه‌ها

    # افزایش وضوح (upsample) برای سطح هموارتر
    if upsample > 1:
        depth = zoom(depth, upsample, order=2)  # bicubic
        # بزرگ‌کردن تصویر برای هماهنگی ابعاد
        # ابعاد جدید
        new_h, new_w = depth.shape
        # بزرگ‌کردن تصویر اصلی برای رنگ‌ها
        from PIL import Image as PILImage
        img_big = img.resize((new_w, new_h), PILImage.BICUBIC)
        pixels = np.array(img_big, dtype=np.float32) / 255.0
        width, height = new_w, new_h

    # مقیاس‌دهی فیزیکی
    max_dim = max(width, height)
    scale_x = 100.0 / max_dim
    scale_y = 100.0 / max_dim
    scale_z = 40.0

    x = np.arange(width) * scale_x
    y = np.arange(height) * scale_y
    xx, yy = np.meshgrid(x, y)
    zz = depth * scale_z

    # رئوس
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
    faces = np.array(faces)

    # محاسبه نرمال‌های رأسی (میانگین‌گیری نرمال وجه‌های همسایه)
    normals_per_vertex = np.zeros_like(vertices)
    count = np.zeros(len(vertices), dtype=int)
    for tri in faces:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in tri:
            normals_per_vertex[idx] += n
            count[idx] += 1
    mask = count > 0
    normals_per_vertex[mask] /= count[mask, np.newaxis]
    normals_per_vertex[~mask] = np.array([0, 0, 1])  # fallback

    # نوشتن OBJ با نرمال‌ها و رنگ‌ها
    lines = ["# Refrigitz Olympic Smooth 3D Relief"]
    for v, c, n in zip(vertices, colors, normals_per_vertex):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        # OBJ format: f v1//vn1 v2//vn2 v3//vn3 (vertex & normal indices same)
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')
