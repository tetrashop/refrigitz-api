import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

def generate_obj(img, smooth_sigma=3.0, upsample=2):
    """
    تولید مش سه‌بعدی با سطح کاملاً هموار و طبیعی:
    - عمق = روشنایی تصویر با هموارسازی قوی (حذف نویز و لبه‌های تیز)
    - افزایش تراکم نقاط با درون‌یابی (upsample)
    - محاسبه نرمال‌های رأسی برای سایه‌زنی صاف
    """
    img = img.convert('RGB')
    width, height = img.size
    # رزولوشن پایه (قبل از upsample)
    base_res = 60
    if width > base_res or height > base_res:
        ratio = min(base_res / width, base_res / height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    pixels = np.array(img, dtype=np.float32) / 255.0

    # محاسبه روشنایی (Luminance)
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

    # هموارسازی قوی نقشهٔ عمق (حذف نوسانات تیز)
    depth = gaussian_filter(gray, sigma=smooth_sigma)

    # نرمال‌سازی مجدد به بازهٔ [0,1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    # افزایش وضوح برای سطح صاف‌تر (درون‌یابی بیکوبیک)
    if upsample > 1:
        depth = zoom(depth, upsample, order=2)  # bicubic
        new_h, new_w = depth.shape
        # بزرگ‌کردن تصویر اصلی برای رنگ‌ها
        img_big = img.resize((new_w, new_h), Image.BICUBIC)
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
    faces = np.array(faces)

    # محاسبه نرمال‌های رأسی (میانگین‌گیری از وجه‌های همسایه)
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
    normals_per_vertex[~mask] = np.array([0, 0, 1])

    # نوشتن OBJ با نرمال‌ها و رنگ‌ها
    lines = ["# Refrigitz Olympic Smooth Relief (No spikes)"]
    for v, c, n in zip(vertices, colors, normals_per_vertex):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')
