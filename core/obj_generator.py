import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_obj(img, target_density=0.15, blur_sigma=2.0, detail_strength=0.45):
    """
    تولید مش سه‌بعدی با تراکم یکنواخت (grid sampling) و نقش برجستهٔ طبیعی.
    target_density: نسبت تعداد نقاط شبکه به max(width,height) (پیش‌فرض 0.15)
    """
    img = img.convert('RGB')
    width, height = img.size

    # ۱. محاسبهٔ عمق (همان Unsharp Masking)
    base_res = 100  # وضوح داخلی برای محاسبهٔ عمق
    if width > base_res or height > base_res:
        ratio = min(base_res / width, base_res / height)
        w_small, h_small = int(width * ratio), int(height * ratio)
        img_small = img.resize((w_small, h_small), Image.LANCZOS)
    else:
        img_small = img.copy()
        w_small, h_small = width, height

    pixels_small = np.array(img_small, dtype=np.float32) / 255.0
    gray = 0.299 * pixels_small[:,:,0] + 0.587 * pixels_small[:,:,1] + 0.114 * pixels_small[:,:,2]
    blurred = gaussian_filter(gray, sigma=blur_sigma)
    detail = gray - blurred
    detail_smooth = gaussian_filter(detail, sigma=0.5)
    depth_small = blurred + detail_smooth * detail_strength
    depth_small = (depth_small - depth_small.min()) / (depth_small.max() - depth_small.min() + 1e-9)

    # ۲. تعیین ابعاد شبکهٔ خروجی بر اساس target_density
    max_dim = max(width, height)
    grid_size = max(5, int(max_dim * target_density))  # تعداد نقاط در هر بُعد
    grid_x = np.linspace(0, width - 1, grid_size)
    grid_y = np.linspace(0, height - 1, grid_size)
    xx, yy = np.meshgrid(grid_x, grid_y)

    # ۳. نگاشت مختصات شبکه به تصویر کوچک (برای نمونه‌برداری عمق)
    # تبدیل مختصات شبکه به اندیس‌های تصویر کوچک
    map_x = xx * (w_small - 1) / (width - 1)
    map_y = yy * (h_small - 1) / (height - 1)
    # نمونه‌برداری خطی از عمق
    from scipy.ndimage import map_coordinates
    depth_grid = map_coordinates(depth_small, [map_y, map_x], order=1, mode='nearest')

    # ۴. نمونه‌برداری رنگ از تصویر اصلی (همان اندازه اصلی)
    # تصویر اصلی را مستقیماً به اندازه شبکه resize می‌کنیم
    img_grid = img.resize((grid_size, grid_size), Image.LANCZOS)
    pixels_grid = np.array(img_grid, dtype=np.float32) / 255.0

    # ۵. مقیاس فیزیکی
    scale_x = 100.0 / max(width, height) * (width / grid_size)
    scale_y = 100.0 / max(width, height) * (height / grid_size)
    scale_z = 40.0

    # مختصات فیزیکی
    x = np.arange(grid_size) * scale_x * (width / grid_size)  # تنظیم مقیاس
    y = np.arange(grid_size) * scale_y * (height / grid_size)
    # ساده‌سازی: یک شبکه یکنواخت در فضای [0,100]x[0,100]
    x = np.linspace(0, 100, grid_size)
    y = np.linspace(0, 100, grid_size)
    xx, yy = np.meshgrid(x, y)
    zz = depth_grid * scale_z

    # رئوس و رنگ‌ها
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    colors = pixels_grid.reshape(-1, 3)

    # ۶. مثلث‌بندی منظم (بدون نیاز به Delaunay)
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            idx0 = i * grid_size + j
            idx1 = i * grid_size + (j + 1)
            idx2 = (i + 1) * grid_size + j
            idx3 = (i + 1) * grid_size + (j + 1)
            faces.append((idx0, idx1, idx3))
            faces.append((idx0, idx3, idx2))
    faces = np.array(faces)

    # ۷. نرمال‌های رأسی
    normals = np.zeros_like(vertices)
    count = np.zeros(len(vertices), dtype=int)
    for tri in faces:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in tri:
            normals[idx] += n
            count[idx] += 1
    mask = count > 0
    normals[mask] /= count[mask, np.newaxis]
    normals[~mask] = np.array([0, 0, 1])

    # ۸. نوشتن OBJ
    lines = ["# Refrigitz Olympic Uniform Mesh"]
    for v, c, n in zip(vertices, colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')
