from waitress import serve
import json

# Upload file
import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import  logging
from pneu_detector import *

pneumonia_model_path = '../ml_model/finalModel/pneu_detect_cnn_model.h5'

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 4*1024 * 1024 # Should not exceed 4MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.jpeg', '.png', '.gif'] # Allowed extensions
app.config['UPLOAD_PATH'] = 'uploads' # uploaded imges path

model= PneumoniaDetector(pneumonia_model_path)


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    im_format = imghdr.what(None, header)
    if not im_format:
        return None
    return '.' + (im_format if im_format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large():
    return "File is too large", 413


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictions')
def predictions():
    return render_template('predictions.html')


@app.route('/apitest')
def apitest(self):
    """check that the API is working"""
    return "API working"


# main API code
@app.route('/', methods=['POST'])
def request_response():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file_copy = uploaded_file
        print("here")
        prediction_dict, file_name = model.check_pneumonia(uploaded_file,filename)
        prediction_json = json.dumps(prediction_dict, ensure_ascii=False)
        file_complete_name= file_name+ file_ext
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], file_complete_name))
    return redirect(url_for('predictions'))


if __name__ == "__main__":
    #serve(app, host="127.0.0.1", port=5005)
    app.run(host="127.0.0.1", debug=True, port=5005)