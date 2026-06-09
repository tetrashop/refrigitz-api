import numpy as np
from PIL import Image

def generate_obj(img):
    """
    تولید مش سه‌بعدی تخت با جابه‌جایی عمق (Displacement Map)
    - نسبت ابعاد کاملاً حفظ می‌شود
    - عمق بر اساس روشنایی (Luminance) محاسبه می‌شود
    - هر رأس رنگ پیکسل خود را دارد (Vertex Color)
    """
    img = img.convert('RGB')
    width, height = img.size
    # حداکثر رزولوشن مش
    max_res = 100
    if width > max_res or height > max_res:
        ratio = min(max_res / width, max_res / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.LANCZOS)

    pixels = np.array(img, dtype=np.float32) / 255.0
    gray = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

    # مقیاس‌بندی: طول و عرض تصویر در بازهٔ ۰ تا ۱۰۰ (یا هر عدد دلخواه)
    scale_x = 100.0 / max(width, height)
    scale_y = 100.0 / max(width, height)
    scale_z = 40.0  # عمق متناسب با ابعاد

    # شبکهٔ نقاط
    x = np.arange(width) * scale_x
    y = np.arange(height) * scale_y
    xx, yy = np.meshgrid(x, y)
    zz = gray * scale_z  # ارتفاع از ۰ تا scale_z

    # رئوس (x, y, z) و رنگ‌ها (r,g,b)
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    colors = pixels.reshape(-1, 3)

    # تولید مثلث‌ها (هر چهارگوش ← دو مثلث)
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            idx0 = i * width + j
            idx1 = i * width + (j + 1)
            idx2 = (i + 1) * width + j
            idx3 = (i + 1) * width + (j + 1)
            faces.append((idx0, idx1, idx3))
            faces.append((idx0, idx3, idx2))

    # نوشتن OBJ با رنگ رأسی (v x y z r g b)
    lines = ["# Refrigitz Olympic 3D Relief Model"]
    for v, c in zip(vertices, colors):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    return "\n".join(lines).encode('utf-8')
