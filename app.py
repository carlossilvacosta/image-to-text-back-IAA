from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, FlaxVisionEncoderDecoderModel
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

loc = "ydshieh/vit-gpt2-coco-en"
feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = FlaxVisionEncoderDecoderModel.from_pretrained(loc)

def generate_step(pixel_values):
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4).sequences
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

@app.route('/describe-image', methods=['POST'])
def describe_image():
    logging.info("Recebendo requisição por /describe-image")
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        with Image.open(requests.get(url, stream=True).raw) as img:
            pixel_values = feature_extractor(images=img, return_tensors="np").pixel_values
        preds = generate_step(pixel_values)
        return jsonify({"description": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
