from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from io import BytesIO
from modules.ocr.ocr import OCRProcessor
from modules.translator import translate
from modules.detection import TextBlockDetector
from modules.utils.translator_utils import format_translations
from modules.rendering.render import draw_text, get_best_render_area


# Hàm làm mờ các vùng chứa chữ
def blur_text_regions(image, blk_list, ksize=(91, 91)):
    image_blurred = image.copy()

    for blk in blk_list:
        if hasattr(blk, 'xyxy') and blk.xyxy is not None and len(blk.xyxy) == 4:
            x1, y1, x2, y2 = map(int, blk.xyxy)
            roi = image_blurred[y1:y2, x1:x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, ksize, 0)
                image_blurred[y1:y2, x1:x2] = blurred_roi

    return image_blurred

def mask_text_regions_white(image, blk_list):
    image_masked = image.copy()

    for blk in blk_list:
        if hasattr(blk, 'xyxy') and blk.xyxy is not None and len(blk.xyxy) == 4:
            x1, y1, x2, y2 = map(int, blk.xyxy)
            if x2 > x1 and y2 > y1:
                image_masked[y1:y2, x1:x2] = 255

    return image_masked

# ========================================
# Thiết lập ngôn ngữ và cấu hình
language_codes = {
    "zh": "Chinese",
    "en": "English",
    "vi": "Vietnamese",
}
alignment = "center"
text_rendering_settings = {
    'font': 'MTOAstroCity.ttf',
    'alignment': 'Center',
    'min_font_size': 12,
    'max_font_size': 40,
    'color': '#000000',
    'upper_case': True,
    'outline': True
}

app = Flask(__name__)
CORS(app)

@app.route('/api/translate/images', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    # Lấy source_lng_cd và target_lng_cd từ yêu cầu
    image_file = request.files['image']
    source_lng_cd = request.form.get('source_lng_cd', 'en')
    target_lng_cd = request.form.get('target_lng_cd', 'vi')

    # Lấy tên ngôn ngữ từ mã ngôn ngữ
    source_lang = language_codes.get(source_lng_cd, 'English')
    target_lang = language_codes.get(target_lng_cd, 'Vietnamese')

    try:
        # Đọc ảnh từ yêu cầu
        in_memory_file = BytesIO(image_file.read())
        file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)

        # Chuyển đổi sang RGB
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Khởi tạo detector
        block_detector = TextBlockDetector(
            'models/detection/comic-speech-bubble-detector.pt',
            'models/detection/comic-text-segmenter.pt',
            'models/detection/manga-text-detector.pt',
            'cpu'
        )
        blk_list = block_detector.detect(image_bgr)

        ocr = OCRProcessor()
        ocr.initialize(source_lang, source_lng_cd)
        blk_list = ocr.process(image_bgr, blk_list)

        # ====================
        # Translate
        blk_list = translate(source_lang, target_lang, blk_list)

        # Tạo folder result nếu chưa có
        # os.makedirs('result', exist_ok=True)

        format_translations(blk_list, target_lng_cd, upper_case=text_rendering_settings['upper_case'])
        
        font_path = f'fonts/{text_rendering_settings["font"]}'
        font_color = text_rendering_settings['color']
        outline = text_rendering_settings['outline']

        # Căn lề cho các text block
        for blk in blk_list:
            blk.alignment = alignment

        # Làm mờ các vùng chứa text
        # blurred_image = blur_text_regions(image_bgr, blk_list)
        blurred_image = mask_text_regions_white(image_bgr, blk_list)

        # Render text dịch lên ảnh đã làm mờ
        rendered_image = draw_text(
            blurred_image, blk_list, font_path,
            colour=font_color,
            init_font_size=text_rendering_settings['max_font_size'],
            min_font_size=text_rendering_settings['min_font_size'],
            outline=outline
        )

        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

        # Chuyển đổi ảnh về định dạng có thể trả về
        _, img_encoded = cv2.imencode('.jpg', rendered_image)
        img_bytes = img_encoded.tobytes()
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='processed_image.jpg'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)