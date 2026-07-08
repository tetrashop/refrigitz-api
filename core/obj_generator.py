import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, ConvexHull
from skimage.restoration import denoise_bilateral

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=60, base_grid=100):
    """
    مدل سه‌بعدی توخالی با دیواره‌های جانبی (Watertight Hollow Shell).
    - نرمال‌ها همگی به سمت بیرون حجم هستند.
    """
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width, height = img_gray.size
    
    max_dim = 60
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img_gray = img_gray.resize((width, height), Image.LANCZOS)
        img_rgb = img_rgb.resize((width, height), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # ۱. بازسازی سطح
    gray_smooth = denoise_bilateral(gray, sigma_color=0.1, sigma_spatial=1.5, channel_axis=None)
    dzdx, dzdy = np.gradient(gray_smooth)
    h, w = gray.shape
    zx = np.cumsum(dzdx, axis=1)
    zy = np.cumsum(dzdy, axis=0)
    depth_map = (zx + zy) / 2.0
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9)
    if invert:
        depth_map = 1.0 - depth_map

    # ۲. نمونه‌برداری تطبیقی
    xx_base = np.tile(np.linspace(0, w-1, base_grid), base_grid)
    yy_base = np.repeat(np.linspace(0, h-1, base_grid), base_grid)
    from scipy.ndimage import map_coordinates
    coords = np.vstack([yy_base, xx_base])
    weight_on_base = np.abs(map_coordinates(depth_map, coords, order=1, mode='nearest')) + 0.1
    probs = weight_on_base / weight_on_base.sum()

    n_select = grid_res * grid_res
    indices = np.random.choice(base_grid * base_grid, size=n_select, replace=False, p=probs)
    chosen_y = yy_base[indices]
    chosen_x = xx_base[indices]

    points_2d = np.column_stack([chosen_x, chosen_y])
    tri = Delaunay(points_2d)

    # ۳. محاسبهٔ عمق
    z_front = depth_map[(np.clip(chosen_y.astype(int), 0, h-1),
                         np.clip(chosen_x.astype(int), 0, w-1))] * height_scale
    shell_thickness = height_scale * 0.08
    z_back = z_front - shell_thickness

    # ۴. ترکیب رئوس و رنگ‌ها
    vertices_front = np.column_stack([chosen_x / w * 100.0, chosen_y / h * 100.0, z_front])
    vertices_back  = np.column_stack([chosen_x / w * 100.0, chosen_y / h * 100.0, z_back])
    all_vertices = np.vstack([vertices_front, vertices_back])

    img_resized = img_rgb.resize((base_grid, base_grid), Image.LANCZOS)
    colors = np.array(img_resized, dtype=np.float32) / 255.0
    colors_flat = colors.reshape(-1, 3)[indices]
    all_colors = np.vstack([colors_flat, colors_flat])

    # ۵. مثلث‌های اصلی (جلو و پشت)
    faces_front = tri.simplices
    offset = len(vertices_front)
    faces_back = tri.simplices + offset
    faces_back = faces_back[:, ::-1]  # معکوس‌سازی برای نرمال صحیح

    # ۶. دوخت لبه‌ها (Edge Stitching) – ایجاد دیواره‌های جانبی
    # پیدا کردن لبه‌های محدب (Convex Hull) از نقاط دوبعدی
    hull = ConvexHull(points_2d)
    side_faces = []
    for simplex in hull.simplices:
        # هر یال محدب یک دیواره مثلثی (دو مثلث) می‌سازد
        i, j = simplex[0], simplex[1]
        # جلو
        v0_front = i
        v1_front = j
        # پشت
        v0_back = i + offset
        v1_back = j + offset
        # دو مثلث برای هر دیواره
        side_faces.append([v0_front, v1_front, v0_back])
        side_faces.append([v1_front, v1_back, v0_back])
    side_faces = np.array(side_faces)

    all_faces = np.vstack([faces_front, faces_back, side_faces])

    # ۷. محاسبهٔ نرمال‌های رأسی (میانگین‌گیری از همهٔ وجه‌های متصل)
    normals = np.zeros_like(all_vertices)
    for simplex in all_faces:
        v0, v1, v2 = all_vertices[simplex[0]], all_vertices[simplex[1]], all_vertices[simplex[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n) + 1e-9)
        for idx in simplex:
            normals[idx] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0, 0, 1])

    # ۸. نوشتن فایل OBJ
    lines = ["# Refrigitz Olympic Hollow Shell (Watertight)"]
    for v, c, n in zip(all_vertices, all_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for simplex in all_faces:
        lines.append(f"f {simplex[0]+1}//{simplex[0]+1} {simplex[1]+1}//{simplex[1]+1} {simplex[2]+1}//{simplex[2]+1}")

    return "\n".join(lines).encode('utf-8')
