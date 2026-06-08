from flask import Flask, request, jsonify, send_file
from io import BytesIO
from PIL import Image
from core._2d_to_3d import process_image_2d_to_3d
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

if __name__ == '__main__':
    app.run()
