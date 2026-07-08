import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=60):
    """
    مدل توخالی پایدار با ضخامت ثابت، شبکهٔ منظم و پردازش فوق‌سریع.
    """
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width = height = grid_res  # مستقیماً روی شبکهٔ نهایی کار می‌کنیم
    img_gray = img_gray.resize((grid_res, grid_res), Image.LANCZOS)
    img_rgb  = img_rgb.resize((grid_res, grid_res), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0

    # عمق صاف و سریع (Unsharp Masking ملایم)
    blurred = gaussian_filter(gray, sigma=3.0)
    detail  = gray - blurred
    depth   = blurred + gaussian_filter(detail, sigma=0.8) * 0.25
    depth   = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    if invert:
        depth = 1.0 - depth

    # شبکهٔ یکنواخت
    x = np.linspace(0, 100, width)
    y = np.linspace(0, 100, height)
    xx, yy = np.meshgrid(x, y)
    z_front = depth * height_scale

    # ضخامت ثابت (پوسته‌ای یکدست)
    thickness = height_scale * 0.08
    z_back = z_front - thickness

    vertices_front = np.stack([xx, yy, z_front], axis=-1).reshape(-1, 3)
    vertices_back  = np.stack([xx, yy, z_back],  axis=-1).reshape(-1, 3)
    all_vertices = np.vstack([vertices_front, vertices_back])

    colors = np.array(img_rgb, dtype=np.float32) / 255.0
    colors_flat = colors.reshape(-1, 3)
    all_colors = np.vstack([colors_flat, colors_flat])

    # مثلث‌بندی جلو و پشت
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
    faces_back = faces_front[:, ::-1] + offset

    # دیواره‌های جانبی (چهار نوار)
    side_faces = []
    # بالا
    for j in range(width - 1):
        a, b = j, j+1
        c, d = a+offset, b+offset
        side_faces.extend([[a,b,d],[a,d,c]])
    # پایین
    base = (height-1)*width
    for j in range(width - 1):
        a, b = base+j, base+j+1
        c, d = a+offset, b+offset
        side_faces.extend([[a,b,d],[a,d,c]])
    # چپ
    for i in range(height - 1):
        a, b = i*width, (i+1)*width
        c, d = a+offset, b+offset
        side_faces.extend([[a,b,d],[a,d,c]])
    # راست
    for i in range(height - 1):
        a, b = i*width+(width-1), (i+1)*width+(width-1)
        c, d = a+offset, b+offset
        side_faces.extend([[a,b,d],[a,d,c]])

    all_faces = np.vstack([faces_front, faces_back, side_faces])

    # نرمال‌های رأسی
    normals = np.zeros_like(all_vertices)
    for tri in all_faces:
        v0, v1, v2 = all_vertices[tri[0]], all_vertices[tri[1]], all_vertices[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n /= (np.linalg.norm(n) + 1e-9)
        normals[tri] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0,0,1])

    # فایل OBJ
    lines = ["# Refrigitz Olympic Stable Hollow Shell"]
    for v, c, n in zip(all_vertices, all_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for tri in all_faces:
        lines.append(f"f {tri[0]+1}//{tri[0]+1} {tri[1]+1}//{tri[1]+1} {tri[2]+1}//{tri[2]+1}")

    return "\n".join(lines).encode('utf-8')
