from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import base64
import traceback

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print(tb)
    response = {'error': str(e)}
    return jsonify(response), 500

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/2d-to-3d', methods=['POST'])
def api_2d_to_3d():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_bytes))
        # کاهش بسیار زیاد اندازه برای جلوگیری از timeout (۲۰۰px)
        img.thumbnail((200, 200))
        # حالت ساده: فقط تصویر کوچک شده را برگردان
        buf = BytesIO()
        img.save(buf, 'PNG')
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
