from waitress import serve
import json

import imghdr
import jsonify
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
import  logging
from pneu_detector import *

API_USED=True
pneumonia_model_path = '../ml_model/finalModel/pneu_detect_cnn_model.h5'

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1000*1024*1024 # Should not exceed 1GB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.jpeg', '.png'] # Allowed extensions
app.config['UPLOAD_PATH'] = 'uploads' # uploaded images path
app.jinja_env.filters['zip'] = zip

# SWAGGER
upload_form_route='/not_save__files' if API_USED else '/save_files'

model= PneumoniaDetector(pneumonia_model_path)


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    im_format = imghdr.what(None, header)
    if not im_format:
        return None
    return '.' + (im_format if im_format != 'jpeg' else 'jpg')

def delete_uploads_folder_content(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error('Failed to delete %s. Reason: %s' % (file_path, e))


@app.route('/apitest', methods=['GET'])
def apitest():
    """check that the API is working"""
    return "API working"

@app.errorhandler(413)
def too_large():
    error_msg="Uploaded files are too large"
    if API_USED:
        return error_msg
    return render_template("file_properties_error_handler.html", config= app.config, error=error_msg),400

@app.errorhandler(BadRequest)
def handle_bad_request():
    bad_request_msg = "This is a bad request"
    if API_USED:
        return bad_request_msg
    return render_template("file_properties_error_handler.html", config= app.config, error=bad_request_msg), BadRequest

@app.route('/properties_error')
def file_type_error():
    error_msg= "Wrong file type"
    if API_USED:
        error_msg+= f". Please use: {','.join(app.config['UPLOAD_EXTENSIONS'])} files !"
        return error_msg
    return render_template("file_properties_error_handler.html", config= app.config, error=error_msg)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/')
def index():
    if API_USED:
        return "This is the Index Page"
    return render_template('index.html')

@app.route(upload_form_route)
def upload_form():
    return render_template('upload_form.html')


@app.route('/save_files', methods=['POST'])
def save_files():
    """save files in a temp folder after deleting its content"""
    delete_uploads_folder_content(app.config['UPLOAD_PATH'])
    uploaded_files = request.files.getlist('files[]')
    logging.warning("\n*******************************************************"
                  f"\nuploaded:{uploaded_files},uploaded_files= {len(uploaded_files)} "
                  "\n*******************************************************")
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        logging.error("\n*******************************************************"
                      f"\nuploaded:{uploaded_file},uploaded_files= {uploaded_files} "
                      "\n*******************************************************")
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            validation_ext = validate_image(uploaded_file.stream)
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] and \
                    file_ext.lower() != validation_ext:
                logging.error("\n*******************************************************"
                                f"\nUnable to upload the specified file: file_ext= {file_ext},validation_ext= {validation_ext}"
                              "\n*******************************************************")
                return  redirect(url_for("file_type_error"))
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    if API_USED:
        return "Files successfully uploaded in temp folder"
    return redirect(url_for('make_predictions'))


@app.route('/predictions')
def make_predictions():
    prediction_dict_list = []
    file_names_list= []
    prediction_dict_for_api = {}
    with os.scandir(app.config['UPLOAD_PATH']) as uploaded_images:
        for image in uploaded_images:
            prediction_dict = model.check_pneumonia(image.path)
            logging.warning("\n*******************************************************"
                        "\nmodel.check_pneumonia() successfully called"
                         f"\nfilename: {image.name}, prediction_dict: {prediction_dict}\n"
                        "\n*******************************************************\n")
            prediction_dict_list.append(prediction_dict)
            logging.warning(prediction_dict_for_api)
            file_names_list.append(image.name)
            if API_USED:
                prediction_dict_for_api[image.name] = prediction_dict
    if API_USED:
        return prediction_dict_for_api # jsonify([json.dumps(prediction) for prediction in prediction_dict_list], ensure_ascii=False)
    return render_template('predictions.html', predictions_infos= prediction_dict_list, filenames=file_names_list)

if __name__ == "__main__":
    #serve(app, host="127.0.0.1", port=5005)
    app.run(host="127.0.0.1", debug=True, port=5005)