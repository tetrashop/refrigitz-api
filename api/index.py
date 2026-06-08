import json
import base64
from io import BytesIO
from werkzeug.wrappers import Request, Response
from PIL import Image
from core.converter import process_image_2d_to_3d
from core.shape_drawing import draw_shape

def api_2d_to_3d(request):
    data = request.get_json()
    if not data or 'image' not in data:
        return Response(json.dumps({'error': 'No image (base64) provided'}), status=400, mimetype='application/json')
    try:
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_bytes))
        result = process_image_2d_to_3d(img)
        buf = BytesIO()
        result.save(buf, 'PNG')
        buf.seek(0)
        return Response(buf.read(), mimetype='image/png')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')

def api_draw_shape(request):
    data = request.get_json()
    if not data:
        return Response(json.dumps({'error': 'JSON required'}), status=400, mimetype='application/json')
    try:
        img = draw_shape(data)
        buf = BytesIO()
        img.save(buf, 'PNG')
        buf.seek(0)
        return Response(buf.read(), mimetype='image/png')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')

def serve_index():
    with open('index.html', 'r') as f:
        return Response(f.read(), mimetype='text/html')

def handler(event, context):
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    body = event.get('body', '') or ''
    is_base64 = event.get('isBase64Encoded', False)
    if is_base64:
        import base64
        body = base64.b64decode(body).decode('utf-8')
    headers = event.get('headers', {})
    environ = {
        'REQUEST_METHOD': method,
        'PATH_INFO': path,
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '443',
        'wsgi.url_scheme': 'https',
        'wsgi.input': BytesIO(body.encode('utf-8')),
        'CONTENT_LENGTH': str(len(body)),
        'HTTP_CONTENT_TYPE': headers.get('content-type', 'application/json'),
    }
    for k, v in headers.items():
        environ['HTTP_' + k.upper().replace('-', '_')] = v

    req = Request(environ)
    if path == '/' and method == 'GET':
        return serve_index()
    elif path == '/api/2d-to-3d' and method == 'POST':
        return api_2d_to_3d(req)
    elif path == '/api/draw-shape' and method == 'POST':
        return api_draw_shape(req)
    else:
        return Response('Not Found', status=404)
