import numpy as np
from PIL import Image

def gaussian_blur(arr, sigma=2.0):
    """فیلتر گوسی دستی با NumPy – جایگزین scipy"""
    kernel_size = int(2 * sigma) | 1  # فرد
    x = np.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
    g = np.exp(-x**2 / (2. * sigma**2))
    g /= g.sum()
    # اعمال جداگانه روی سطرها و ستون‌ها
    tmp = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 1, arr)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode='same'), 0, tmp)

def generate_obj(img_pil, invert=False, height_scale=40.0, grid_res=40, floor_z=0.0):
    img_rgb = img_pil.convert('RGB')
    img_gray = img_rgb.convert('L')
    width = height = grid_res
    img_gray = img_gray.resize((grid_res, grid_res), Image.LANCZOS)
    img_rgb  = img_rgb.resize((grid_res, grid_res), Image.LANCZOS)

    gray = np.array(img_gray, dtype=np.float32) / 255.0
    blurred = gaussian_blur(gray, sigma=1.5)
    depth   = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-9)
    if invert:
        depth = 1.0 - depth

    x = np.linspace(0, 100, width)
    y = np.linspace(0, 100, height)
    xx, yy = np.meshgrid(x, y)
    z_front = depth * height_scale
    z_back  = np.where(floor_z is not None, floor_z, 2 * np.mean(z_front) - z_front)

    vertices_front = np.stack([xx, yy, z_front], axis=-1).reshape(-1, 3)
    vertices_back  = np.stack([xx, yy, z_back],  axis=-1).reshape(-1, 3)
    all_vertices = np.vstack([vertices_front, vertices_back])

    colors = np.array(img_rgb, dtype=np.float32) / 255.0
    all_colors = np.vstack([colors.reshape(-1, 3), colors.reshape(-1, 3)])

    faces_front = []
    for i in range(height - 1):
        for j in range(width - 1):
            a = i * width + j
            b = a + 1
            c = (i + 1) * width + j
            d = c + 1
            faces_front.extend([[a,b,d],[a,d,c]])
    faces_front = np.array(faces_front, dtype=int)

    offset = len(vertices_front)
    faces_back = faces_front[:, ::-1] + offset

    side_faces = []
    for j in range(width - 1):
        side_faces.extend([[j,j+1,j+offset+1],[j,j+offset+1,j+offset]])
    base = (height-1)*width
    for j in range(width - 1):
        side_faces.extend([[base+j,base+j+1,base+j+offset+1],[base+j,base+j+offset+1,base+j+offset]])
    for i in range(height - 1):
        side_faces.extend([[i*width,(i+1)*width,(i+1)*width+offset],[i*width,(i+1)*width+offset,i*width+offset]])
    for i in range(height - 1):
        side_faces.extend([[i*width+(width-1),(i+1)*width+(width-1),(i+1)*width+(width-1)+offset],
                           [i*width+(width-1),(i+1)*width+(width-1)+offset,i*width+(width-1)+offset]])
    all_faces = np.vstack([faces_front, faces_back, side_faces])

    normals = np.zeros_like(all_vertices)
    for tri in all_faces:
        v0, v1, v2 = all_vertices[tri[0]], all_vertices[tri[1]], all_vertices[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n /= (np.linalg.norm(n) + 1e-9)
        normals[tri] += n
    mask = np.linalg.norm(normals, axis=1) > 0
    normals[mask] /= np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[~mask] = np.array([0,0,1])

    lines = ["# Refrigitz Ultra-Fast Hollow Shell"]
    for v, c, n in zip(all_vertices, all_colors, normals):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for tri in all_faces:
        lines.append(f"f {tri[0]+1}//{tri[0]+1} {tri[1]+1}//{tri[1]+1} {tri[2]+1}//{tri[2]+1}")

    return "\n".join(lines).encode('utf-8')
