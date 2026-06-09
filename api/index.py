from flask import Flask, request, jsonify, send_file
from io import BytesIO
from PIL import Image
import base64
import traceback
from core.converter import process_image_2d_to_3d
from core.shape_drawing import draw_shape

app = Flask(__name__)

# مدیریت خطای سراسری: هر استثنا → JSON
@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print(tb)  # در محیط production می‌توانید لاگ کنید
    response = {
        'error': str(e),
        'type': type(e).__name__
    }
    # اگر خطا از نوع HTTP باشد، کد وضعیت آن را برمی‌گردانیم
    if hasattr(e, 'code'):
        return jsonify(response), e.code
    return jsonify(response), 500

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return jsonify({'error': 'Could not load interface', 'detail': str(e)}), 500

@app.route('/api/2d-to-3d', methods=['POST'])
def api_2d_to_3d():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image (base64) provided'}), 400
    try:
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_bytes))
        result = process_image_2d_to_3d(img)
        buf = BytesIO()
        result.save(buf, 'PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        # بازگشت JSON خطا
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route('/api/draw-shape', methods=['POST'])
def api_draw_shape():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400
    try:
        img = draw_shape(data)
        buf = BytesIO()
        img.save(buf, 'PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500
