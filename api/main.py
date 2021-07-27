from flask import Flask, request, jsonify
from api.torch_utils import transform_image, get_prediction
import pandas as pd

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            temp_df = pd.read_csv('api/MetaData.csv')
            ind = prediction.item()
            data = {'prediction': prediction.item(), 'class_name': str(temp_df.iloc[ind,1])}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})