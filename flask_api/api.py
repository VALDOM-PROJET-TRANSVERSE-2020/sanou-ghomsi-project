from tensorflow.keras.models import load_model
import numpy as np
from waitress import serve
import json

# Upload file
import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename

pneumonia_model_path = '../ml_model/finalModel/pneu_detect_cnn_model.h5'

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 4*1024 * 1024 # Should not exceed 4MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif'] # Allowed extensions
app.config['UPLOAD_PATH'] = 'uploads' # uploaded imges path


class PneumoniaDetector(object):

    def __init__(self, model_path):
        self.prd_model = load_model(model_path)

    @staticmethod
    def __validate_image(stream):
        header = stream.read(512)
        stream.seek(0)
        im_format = imghdr.what(None, header)
        if not im_format:
            return None
        return '.' + (im_format if im_format != 'jpeg' else 'jpg')

    @staticmethod
    @app.errorhandler(413)
    def too_large():
        return "File is too large", 413

    @staticmethod
    @app.route('/uploads/<filename>')
    def upload(filename):
        return send_from_directory(app.config['UPLOAD_PATH'], filename)

    @staticmethod
    @app.route('/')
    def index():
        return render_template('index.html')

    @staticmethod
    @app.route('/predictions')
    def predictions():
        return render_template('predictions.html')

    def __predict(self, image_file):
        pred_probas = self.prd_model(pneumonia_model_path).predict(image_file)
        pred_index = np.argmax(pred_probas, axis=1)
        pred_confidence = pred_probas[pred_index]
        return pred_index, pred_confidence

    @app.route('/', methods=['POST'])
    def upload_files(self):
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != self.__validate_image(uploaded_file.stream):
                abort(400)
            prediction_json = self.check_pneumonia(self, filename)
            print(prediction_json)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        return redirect(url_for('predictions'))

    @app.route('/apitest')
    def apitest(self):
        """check that the API is working"""
        return "API working"

    # main API code
    @app.route('/detection', methods=['POST'])
    def check_pneumonia(self, image_file):
        error_message= "Failed to predict the class: \t"
        try:
            has_pneumonia, pred_confidence = self.__predict(image_file)
            if has_pneumonia:
                prediction = "Pneumonia detected"
            else:
                prediction = "Normal Chest"
            prediction_dict = {'pred': prediction, 'proba': pred_confidence}
            prediction_json = json.dumps(prediction_dict, ensure_ascii=False)
        except Exception as ex:
            return error_message + ex
        return prediction_json


if __name__ == "__main__":
    serve(app, host="127.0.0.1", debug=True, port=5005)