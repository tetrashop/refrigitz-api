from PIL import Image, ImageDraw

def draw_shape(data):
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    shape = data.get('shape')
    color = data.get('color', 'black')
    params = data.get('params', [])

    if shape == 'line':
        if len(params) < 4:
            raise ValueError('خط نیاز به ۴ پارامتر (x1,y1,x2,y2) دارد.')
        draw.line([(params[0], params[1]), (params[2], params[3])], fill=color, width=3)
    elif shape == 'arc':
        if len(params) < 6:
            raise ValueError('کمان نیاز به ۶ پارامتر (x1,y1,x2,y2,start,end) دارد.')
        draw.arc([params[0], params[1], params[2], params[3]], start=params[4], end=params[5], fill=color, width=3)
    elif shape == 'bezier':
        if len(params) < 8:
            raise ValueError('بزیه نیاز به ۸ پارامتر (x1,y1,x2,y2,x3,y3,x4,y4) دارد.')
        points = [(params[i], params[i+1]) for i in range(0, len(params), 2)]
        draw.line(points, fill=color, width=3)
    elif shape == 'ellipse':
        if len(params) < 4:
            raise ValueError('بیضی نیاز به ۴ پارامتر (x1,y1,x2,y2) دارد.')
        draw.ellipse([params[0], params[1], params[2], params[3]], outline=color, width=3)
    elif shape == 'rectangle':
        if len(params) < 4:
            raise ValueError('مستطیل نیاز به ۴ پارامتر (x1,y1,x2,y2) دارد.')
        draw.rectangle([params[0], params[1], params[2], params[3]], outline=color, width=3)
    else:
        raise ValueError(f"شکل نامعتبر: {shape}")
    return img
