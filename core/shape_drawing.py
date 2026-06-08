from PIL import Image, ImageDraw

def draw_shape(data):
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    shape = data.get('shape')
    color = data.get('color', 'black')
    params = data.get('params', [])
    if shape == 'line':
        draw.line([(params[0], params[1]), (params[2], params[3])], fill=color, width=3)
    elif shape == 'arc':
        draw.arc([params[0], params[1], params[2], params[3]], start=params[4], end=params[5], fill=color, width=3)
    elif shape == 'bezier':
        points = [(params[i], params[i+1]) for i in range(0, len(params), 2)]
        # منحنی بزیه ساده با استفاده از line (کتابخانه استاندارد)
        draw.line(points, fill=color, width=3)
    elif shape == 'ellipse':
        draw.ellipse([params[0], params[1], params[2], params[3]], outline=color, width=3)
    elif shape == 'rectangle':
        draw.rectangle([params[0], params[1], params[2], params[3]], outline=color, width=3)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    return img
