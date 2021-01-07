#from tensorflow.keras.models import load_model
#import h5py
#import numpy as np
#import pandas as pd
import json

# Upload file
import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

pneumonia_model_path = 'pneu_detect_cnn_model.hdf5'

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


class PneumoniaDetector(object):

    def __init__(self, model_path):
        # self.prd_model = load_model(model_path)
    @staticmethod
    def __validate_image(stream):
        header = stream.read(512)
        stream.seek(0)
        im_format = imghdr.what(None, header)
        if not im_format:
            return None
        return '.' + (im_format if im_format != 'jpeg' else 'jpg')

    @app.route('/')
    def index(self):
        return render_template('index.html')

    @app.route('/predictions')
    def predictions(self):
        return render_template('predictions.html')

    def __predict(self, image_file):
        #pred_probas = self.prd_model.predict(image_file)
        #pred_index = np.argmax(pred_probas, axis=1)
        #pred_confidence = pred_probas[pred_index]
        return 1, 0.83 #pred_index, pred_confidence

    @app.route('/', methods=['POST'])
    def upload_files(self):
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != self.__validate_image(uploaded_file.stream):
                abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        return redirect(url_for('predictions'))

    @app.route('/apitest')
    def apitest(self):
        """check that the API is working"""
        return "API working"

    # main API code
    @app.route('/detection', methods=['POST'])
    def check_pneumonia(self, image_file):
        error_message= "Failed to execute the request: \t"
        try:
            if request.method == 'POST':
                # TODO: How to receive an image via post request
                pass

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
    app.run(host="127.0.0.1", debug=False, port=5005)