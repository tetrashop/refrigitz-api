import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
from skimage.restoration import denoise_bilateral

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=60, base_grid=100):
    """
    مدل سه‌بعدی با سطوح مثلثی مورب (بدون Heightmap ساده)
    - بازسازی سطح با انتگرال‌گیری سریع از گرادیان‌ها (بدون FFT)
    - تراکم تطبیقی رئوس در لبه‌ها
    """
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width, height = img_gray.size
    # کاهش شدید ابعاد برای اجرا در ۱۰ ثانیه
    max_dim = 50
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img_gray = img_gray.resize((width, height), Image.LANCZOS)
        img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۱. پیش‌پردازش و گرادیان‌ها
    gray_smooth = denoise_bilateral(gray, sigma_color=0.1, sigma_spatial=1.5,
                                    channel_axis=None)
    dzdx, dzdy = np.gradient(gray_smooth)

    # ۲. انتگرال‌گیری سریع (بازسازی سطح تقریبی)
    #     با جمع زدن گرادیان‌ها در راستای x و y و میانگین‌گیری
    h, w = gray.shape
    zx = np.cumsum(dzdx, axis=1)
    zy = np.cumsum(dzdy, axis=0)
    # تصحیح میانگین
    depth_map = (zx + zy) / 2.0
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9)

    if invert:
        depth_map = 1.0 - depth_map

    # ۳. نمونه‌برداری تطبیقی (نقاط متراکم‌تر روی لبه‌ها)
    xx_base, yy_base = np.meshgrid(
        np.linspace(0, w-1, base_grid),
        np.linspace(0, h-1, base_grid)
    )
    from scipy.ndimage import map_coordinates
    weight_on_base = np.abs(map_coordinates(depth_map, [yy_base, xx_base], order=1, mode='nearest')) + 0.1
    probs = weight_on_base.flatten() / weight_on_base.sum()

    chosen_indices = np.random.choice(base_grid * base_grid, size=grid_res * grid_res,
                                      replace=False, p=probs)
    chosen_y = yy_base.flatten()[chosen_indices]
    chosen_x = xx_base.flatten()[chosen_indices]

    points_2d = np.column_stack([chosen_x, chosen_y])
    tri = Delaunay(points_2d)

    # ۴. عمق نهایی
    zz = depth_map[(np.clip(chosen_y.astype(int), 0, h-1),
                    np.clip(chosen_x.astype(int), 0, w-1))] * height_scale

    # ۵. رنگ‌ها
    img_resized = img_rgb.resize((base_grid, base_grid), Image.LANCZOS)
    colors = np.array(img_resized, dtype=np.float32) / 255.0
    colors = colors.reshape(-1, 3)[chosen_indices]

    # ۶. مختصات
    vertices = np.column_stack([
        chosen_x / w * 100.0,
        chosen_y / h * 100.0,
        zz
    ])

    # ۷. نرمال‌های رأسی (میانگین نرمال وجه‌های همسایه)
    normals = np.zeros_like(vertices)
    for simplex in tri.simplices:
        v0, v1, v2 = vertices[simplex[0]], vertices[simplex[1]], vertices[simplex[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in simplex:
            normals[idx] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0, 0, 1])

    # ۸. نوشتن OBJ
    lines = ["# Refrigitz Faceted Mesh (Fast Integration)"]
    for v, c, n in zip(vertices, colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for simplex in tri.simplices:
        lines.append(f"f {simplex[0]+1}//{simplex[0]+1} {simplex[1]+1}//{simplex[1]+1} {simplex[2]+1}//{simplex[2]+1}")

    return "\n".join(lines).encode('utf-8')
