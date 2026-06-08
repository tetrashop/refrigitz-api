from io import BytesIO
from flask import Flask, request, jsonify, send_file
from PIL import Image
from core.converter import process_image_2d_to_3d
from core.shape_drawing import draw_shape

app = Flask(__name__)

@app.route('/api/2d-to-3d', methods=['POST'])
def api_2d_to_3d():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        img = Image.open(file.stream)
        result_img = process_image_2d_to_3d(img)
        img_io = BytesIO()
        result_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/draw-shape', methods=['POST'])
def api_draw_shape():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400
    try:
        img = draw_shape(data)
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# handler دستی برای Vercel (بدون نیاز به serverless_wsgi)
from werkzeug.wrappers import Response
import json

def handler(event, context):
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    headers = event.get('headers', {})
    body = event.get('body', '')
    is_base64 = event.get('isBase64Encoded', False)

    if is_base64:
        import base64
        body = base64.b64decode(body)
    else:
        body = body.encode('utf-8')

    environ = {
        'REQUEST_METHOD': method,
        'PATH_INFO': path,
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '443',
        'wsgi.url_scheme': 'https',
        'wsgi.input': BytesIO(body),
        'CONTENT_LENGTH': str(len(body)),
        'HTTP_CONTENT_TYPE': headers.get('content-type', ''),
    }
    for k, v in headers.items():
        key = 'HTTP_' + k.upper().replace('-', '_')
        environ[key] = v

    with app.request_context(environ):
        response = app.full_dispatch_request()
        # تبدیل پاسخ به فرمت Vercel
        resp_headers = {k: v for k, v in response.headers}
        resp_body = response.get_data(as_text=False)
        return {
            'statusCode': response.status_code,
            'headers': resp_headers,
            'body': resp_body.decode('utf-8'),
            'isBase64Encoded': False
        }
