import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, sobel, map_coordinates
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors

def generate_obj(img, detail_strength=0.5, grid_res=80):
    """
    مدل سه‌بعدی با ترکیب شبکهٔ یکنواخت + درون‌یابی خطی روی لبه‌ها.
    """
    img = img.convert('RGB')
    width, height = img.size
    # محدودیت اندازه برای سرعت
    max_dim = 100
    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    pixels = np.array(img, dtype=np.float32) / 255.0
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

    # ---------- 1. نقشهٔ عمق پایه (Unsharp Masking) ----------
    blurred = gaussian_filter(gray, sigma=2.0)
    detail = gray - blurred
    detail_smooth = gaussian_filter(detail, sigma=0.5)
    depth_base = blurred + detail_smooth * detail_strength
    depth_base = (depth_base - depth_base.min()) / (depth_base.max() - depth_base.min() + 1e-9)

    # ---------- 2. تشخیص لبه‌ها (Canny ساده) ----------
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_binary = edge_mag > np.percentile(edge_mag, 80)  # لبه‌های قوی
    # نازک‌سازی (اختیاری)
    from scipy.ndimage import binary_erosion
    edge_binary = binary_erosion(edge_binary, iterations=1)

    # ---------- 3. نقاط لبه با مختصات (x,y,z) ----------
    ys, xs = np.where(edge_binary)
    edge_points = []
    for x, y in zip(xs, ys):
        z = depth_base[y, x]
        edge_points.append((x, y, z))
    edge_points = np.array(edge_points)

    # اگر لبه کافی نبود، از همه نقاط شبکه استفاده کن
    if len(edge_points) < 10:
        # برگشت به روش ساده‌تر
        return _simple_grid_mesh(img, depth_base, grid_res)

    # ---------- 4. برازش خط سه‌بعدی برای هر پیکسل لبه و تولید نقاط جدید ----------
    # برای سرعت، فقط یک‌درمیان نقاط لبه را بررسی کن
    new_points = []
    stride = max(1, len(edge_points) // 200)  # محدود به ۲۰۰ نقطه
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(edge_points)
    for i in range(0, len(edge_points), stride):
        pt = edge_points[i].reshape(1, -1)
        dist, idx = neigh.kneighbors(pt)
        # همسایگان نزدیک
        neighbors = edge_points[idx[0, 1:]]  # ۹ همسایه نزدیک
        # برازش خط با PCA (بردار اصلی)
        if len(neighbors) < 3:
            continue
        mean = neighbors.mean(axis=0)
        centered = neighbors - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = eigenvectors[:, -1]  # بردار با بزرگترین واریانس
        # تولید ۵ نقطه در امتداد خط از pt
        t_vals = np.linspace(-3, 3, 7)  # گام‌های ۰٫۵ پیکسل
        for t in t_vals:
            new_pt = pt[0] + direction * t
            # محدود به ابعاد تصویر
            nx, ny = new_pt[0], new_pt[1]
            if 0 <= nx < width and 0 <= ny < height:
                new_points.append(new_pt)
    if new_points:
        new_points = np.array(new_points)
        # ترکیب با نقاط شبکه یکنواخت (برای grid interpolation)
        # ایجاد شبکه متراکم از همه نقاط
        all_points = np.vstack([edge_points, new_points])
    else:
        all_points = edge_points

    # ---------- 5. تولید نقشهٔ عمق جدید با griddata ----------
    grid_x = np.linspace(0, width-1, grid_res)
    grid_y = np.linspace(0, height-1, grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    depth_grid = griddata(
        (all_points[:, 0], all_points[:, 1]), all_points[:, 2],
        (grid_xx, grid_yy), method='linear', fill_value=0.0
    )
    # پر کردن نقاط خالی با نزدیکترین همسایه
    from scipy.interpolate import NearestNDInterpolator
    nan_mask = np.isnan(depth_grid)
    if np.any(nan_mask):
        interp = NearestNDInterpolator(all_points[:, :2], all_points[:, 2])
        depth_grid[nan_mask] = interp(grid_xx[nan_mask], grid_yy[nan_mask])
    depth_grid = np.clip(depth_grid, 0, 1)

    # ---------- 6. ساخت مش نهایی ----------
    img_resized = img.resize((grid_res, grid_res), Image.LANCZOS)
    colors = np.array(img_resized, dtype=np.float32) / 255.0

    scale_x = 100.0 / max(width, height) * (width / grid_res)
    scale_y = 100.0 / max(width, height) * (height / grid_res)
    scale_z = 40.0

    x = np.linspace(0, 100, grid_res)
    y = np.linspace(0, 100, grid_res)
    xx, yy = np.meshgrid(x, y)
    zz = depth_grid * scale_z

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

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

    lines = ["# Refrigitz Edge-Fitted 3D Model"]
    for v, c, n in zip(vertices, colors.reshape(-1, 3), normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")

    return "\n".join(lines).encode('utf-8')

def _simple_grid_mesh(img, depth_map, grid_res):
    # fallback ساده
    width, height = img.size
    img_rs = img.resize((grid_res, grid_res), Image.LANCZOS)
    pixels = np.array(img_rs, dtype=np.float32) / 255.0
    x = np.linspace(0, 100, grid_res)
    y = np.linspace(0, 100, grid_res)
    xx, yy = np.meshgrid(x, y)
    zz = depth_map * 40.0
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(grid_res-1):
        for j in range(grid_res-1):
            a = i*grid_res + j
            b = a + 1
            c = (i+1)*grid_res + j
            d = c + 1
            faces.append((a,b,d)); faces.append((a,d,c))
    faces = np.array(faces)
    normals = np.zeros_like(vertices)
    cnt = np.zeros(len(vertices), dtype=int)
    for tri in faces:
        v0,v1,v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(v1-v0, v2-v0); n /= (np.linalg.norm(n)+1e-9)
        for idx in tri: normals[idx] += n; cnt[idx] += 1
    mask = cnt>0; normals[mask] /= cnt[mask, np.newaxis]; normals[~mask] = np.array([0,0,1])
    lines = ["# Refrigitz Simple Mesh"]
    for v,c,n in zip(vertices, pixels.reshape(-1,3), normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for f in faces: lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")
    return "\n".join(lines).encode('utf-8')
