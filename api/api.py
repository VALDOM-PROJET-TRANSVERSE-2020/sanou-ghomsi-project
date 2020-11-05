from tensorflow.keras.models import load_model
import h5py
import numpy as np
import pandas as pd
from flask import Flask, url_for, request
import json

pneumonia_model_path = 'pneu_detect_cnn_model.hdf5'

app = Flask(__name__)


class PneumoniaDetector(object):
    def __init__(self, model_path):
        self.prd_model = load_model(model_path)

    @app.route('/apitest')
    def apitest(self):
        """check that the API is working"""
        return "API working"

    def __predict(self, image_file):
        pred_probas = self.prd_model.predict(image_file)
        pred_index = np.argmax(pred_probas, axis=1)
        pred_confidence = pred_probas[pred_index]
        return pred_index, pred_confidence

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
    app.run(host="0.0.0.0", debug=False, port=5005)