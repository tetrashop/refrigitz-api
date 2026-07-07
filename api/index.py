from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import base64
from core.converter import process_image_2d_to_3d
from core.obj_generator import generate_obj

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/2d-to-3d', methods=['POST'])
def api_2d_to_3d():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    fmt = request.args.get('format', 'png')
    invert = request.args.get('invert', 'false').lower() == 'true'
    alpha = float(request.args.get('alpha', 0.85))
    height = float(request.args.get('height', 40.0))  # پیش‌فرض ۴۰ (ارتفاع معمولی)، صفر = صفحه تخت
    try:
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        if fmt == 'obj':
            obj_data = generate_obj(img, invert=invert, alpha=alpha, height_scale=height)
            buf = BytesIO(obj_data)
            resp = make_response(send_file(buf, mimetype='application/octet-stream',
                                           as_attachment=True, download_name='model.obj'))
        else:
            result_img = process_image_2d_to_3d(img)
            buf = BytesIO()
            result_img.save(buf, 'PNG')
            buf.seek(0)
            resp = make_response(send_file(buf, mimetype='image/png'))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/draw-shape', methods=['POST'])
def api_draw_shape():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON required'}), 400
    try:
        from core.shape_drawing import draw_shape
        img = draw_shape(data)
        buf = BytesIO()
        img.save(buf, 'PNG')
        buf.seek(0)
        resp = make_response(send_file(buf, mimetype='image/png'))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
