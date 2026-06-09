import numpy as np
from PIL import Image
from scipy.ndimage import sobel, zoom
from scipy.spatial import Delaunay
from skimage.restoration import denoise_bilateral

def generate_obj(img_pil, invert=False, grid_res=80, base_grid=200):
    """
    مدل سه‌بعدی با نمونه‌برداری تطبیقی (Adaptive Sampling):
    - تراکم رئوس در نواحی لبه (جزئیات) بیشتر است
    - مثلث‌بندی Delaunay برای مش بهینه
    - عمق = Edge Magnitude (جهت‌خنثی)
    """
    # تصویر رنگی (برای رنگ‌های رأسی)
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')

    width, height = img_gray.size
    max_dim = 100  # ابعاد پردازشی
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img_gray = img_gray.resize((width, height), Image.LANCZOS)
        img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۱. فیلتر دوطرفه برای کاهش نویز و حفظ لبه
    gray_smooth = denoise_bilateral(gray, sigma_color=0.05, sigma_spatial=2,
                                    channel_axis=None)

    # ۲. قدرت لبه (Edge Magnitude)
    edges_x = sobel(gray_smooth, axis=0)
    edges_y = sobel(gray_smooth, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-9)

    # ۳. ساخت شبکهٔ پایهٔ متراکم (base_grid × base_grid)
    xx_base, yy_base = np.meshgrid(
        np.linspace(0, width-1, base_grid),
        np.linspace(0, height-1, base_grid)
    )
    # قدرت لبه در هر نقطه (درون‌یابی خطی)
    from scipy.ndimage import map_coordinates
    edge_on_base = map_coordinates(edge_mag, [yy_base, xx_base], order=1, mode='nearest')

    # احتمال انتخاب هر نقطه = 1 + لبه
    probs = 1.0 + edge_on_base.flatten()
    probs /= probs.sum()

    # انتخاب grid_res نقطه به صورت وزن‌دار (بدون جایگذاری)
    chosen_indices = np.random.choice(
        base_grid * base_grid,
        size=grid_res * grid_res,
        replace=False,
        p=probs
    )
    chosen_y = yy_base.flatten()[chosen_indices]
    chosen_x = xx_base.flatten()[chosen_indices]

    # ۴. مثلث‌بندی Delaunay روی نقاط انتخاب‌شده
    points_2d = np.column_stack([chosen_x, chosen_y])
    tri = Delaunay(points_2d)

    # ۵. عمق = Edge Magnitude در هر نقطه (مقیاس‌دهی)
    zz = edge_mag[(np.clip(chosen_y.astype(int), 0, height-1),
                   np.clip(chosen_x.astype(int), 0, width-1))] * 40.0
    if invert:
        zz = 40.0 - zz

    # ۶. رنگ‌ها از تصویر اصلی
    colors = img_rgb.resize((base_grid, base_grid), Image.LANCZOS)
    colors = np.array(colors, dtype=np.float32) / 255.0
    colors = colors.reshape(-1, 3)[chosen_indices]

    # ۷. مقیاس‌دهی X,Y به فضای [0,100]
    vertices = np.column_stack([
        chosen_x / width * 100.0,
        chosen_y / height * 100.0,
        zz
    ])

    # ۸. نرمال‌ها
    normals = np.zeros_like(vertices)
    # برای هر مثلث
    for simplex in tri.simplices:
        v0, v1, v2 = vertices[simplex[0]], vertices[simplex[1]], vertices[simplex[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in simplex:
            normals[idx] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0, 0, 1])

    # ۹. نوشتن OBJ
    lines = ["# Refrigitz Adaptive Mesh (Edge-Importance Sampling)"]
    for v, c, n in zip(vertices, colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for simplex in tri.simplices:
        lines.append(f"f {simplex[0]+1}//{simplex[0]+1} {simplex[1]+1}//{simplex[1]+1} {simplex[2]+1}//{simplex[2]+1}")

    return "\n".join(lines).encode('utf-8')
