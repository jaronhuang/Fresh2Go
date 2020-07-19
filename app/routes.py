from app import app
import os
from flask import Flask, request, redirect, url_for, flash, send_from_directory, current_app, render_template
from script import predictFruitClass, trained_model, class_dict
from werkzeug.utils import secure_filename
import base64

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/index', methods=['GET'])
def index():
    # return "hello"
    # return send_from_directory(app.config['UPLOAD_FOLDER'], 'index.html')
    # return current_app.send_static_file('./index.html')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'no file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return(predictFruitClass(os.path.join(app.config['UPLOAD_FOLDER'], filename), trained_model, class_dict))
            # return "success"
    else: 
        return "wrong method"
    