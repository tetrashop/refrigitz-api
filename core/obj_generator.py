import numpy as np
from PIL import Image
from scipy.ndimage import sobel, zoom
from scipy.spatial import Delaunay
from skimage.restoration import denoise_bilateral

def generate_obj(img_pil, invert=False, alpha=0.85, grid_res=80, base_grid=150):
    """
    مدل سه‌بعدی با ترکیب هوشمند لبه و روشنایی (Edge + Intensity):
    alpha: وزن لبه (پیش‌فرض 0.85) – هرچه بیشتر، جزئیات تیزتر.
    """
    # تصاویر
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width, height = img_gray.size
    max_dim = 100
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img_gray = img_gray.resize((width, height), Image.LANCZOS)
        img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۱. Edge Magnitude (لبه‌ها)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-9)

    # ۲. ترکیب عمق
    depth_map = alpha * edge_mag + (1 - alpha) * gray

    # ۳. فیلتر دوطرفه (حفظ لبه، هموارسازی)
    depth_map = denoise_bilateral(depth_map, sigma_color=0.1, sigma_spatial=2,
                                  channel_axis=None)
    # نرمال‌سازی مجدد
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9)

    if invert:
        depth_map = 1.0 - depth_map

    # ۴. شبکه پایه و احتمال انتخاب نقاط (ترکیب edge+intensity برای شانس بیشتر در نواحی نیمه‌صاف)
    xx_base, yy_base = np.meshgrid(
        np.linspace(0, width-1, base_grid),
        np.linspace(0, height-1, base_grid)
    )
    from scipy.ndimage import map_coordinates
    # نمونه‌برداری از نقشه وزن (همان depth_map) برای احتمال
    weight_on_base = map_coordinates(depth_map, [yy_base, xx_base], order=1, mode='nearest')
    probs = weight_on_base.flatten() + 0.1  # حداقل شانس
    probs /= probs.sum()

    # انتخاب grid_res * grid_res نقطه
    chosen_indices = np.random.choice(
        base_grid * base_grid,
        size=grid_res * grid_res,
        replace=False,
        p=probs
    )
    chosen_y = yy_base.flatten()[chosen_indices]
    chosen_x = xx_base.flatten()[chosen_indices]

    # ۵. مثلث‌بندی Delaunay
    points_2d = np.column_stack([chosen_x, chosen_y])
    tri = Delaunay(points_2d)

    # ۶. عمق نهایی از depth_map اصلی
    zz = depth_map[(np.clip(chosen_y.astype(int), 0, height-1),
                    np.clip(chosen_x.astype(int), 0, width-1))] * 40.0
    if invert:
        zz = 40.0 - zz

    # ۷. رنگ‌ها از تصویر اصلی
    img_resized = img_rgb.resize((base_grid, base_grid), Image.LANCZOS)
    colors = np.array(img_resized, dtype=np.float32) / 255.0
    colors = colors.reshape(-1, 3)[chosen_indices]

    # ۸. مقیاس‌دهی X,Y
    vertices = np.column_stack([
        chosen_x / width * 100.0,
        chosen_y / height * 100.0,
        zz
    ])

    # ۹. نرمال‌ها
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

    # ۱۰. نوشتن OBJ
    lines = ["# Refrigitz Olympic Hybrid Edge+Intensity"]
    for v, c, n in zip(vertices, colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for simplex in tri.simplices:
        lines.append(f"f {simplex[0]+1}//{simplex[0]+1} {simplex[1]+1}//{simplex[1]+1} {simplex[2]+1}//{simplex[2]+1}")

    return "\n".join(lines).encode('utf-8')
