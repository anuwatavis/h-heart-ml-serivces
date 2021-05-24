from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS
import os
import uuid
import requests

from datetime import datetime
app = Flask(__name__)
UPLOAD_FOLDER = '/app/app/csv_upload_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
#model_lgr = load_model(f'./h_jorth_logistic_regression')


@app.route("/")
def hello():
    return 'hello'


@app.route('/predict', methods=['POST'])
def predict():
    print('/predict called')
    file = request.files['ecgFile']
    model_idx = request.form['model_idx']
    print(model_idx)
    model_lgr = load_model('/app/app/' + model_idx)
    print('./ecg_model/' + model_idx)
    filename = file.filename
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S_")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], dt_string + filename))
    df = pd.read_csv(os.path.join(
        app.config['UPLOAD_FOLDER'], dt_string + filename))
    ecg_data = list(df['ecg'])
    activity = calculateActivity(ecg_data)
    mobility = calcMobility(ecg_data)
    complexity = calcComplexity(ecg_data)

    hjorth_feature = pd.DataFrame([[activity, mobility, complexity]], columns=[
        'activity', 'mobility', 'complexity'])
    predictions = predict_model(model_lgr, data=hjorth_feature)
    print(predictions)
    return {
        "statusCode": 200,
        "message": 'Infernce with model selected done.',
        "activity": str(predictions['activity'][0]),
        'mobility': str(predictions['mobility'][0]),
        'complexity': str(predictions['complexity'][0]),
        'label': str(predictions['Label'][0]),
        'score': str(predictions['Score'][0])
    }


@app.route('/model', methods=['GET'])
def model():
    r = requests.get('http://www.google.com')
    print(r)
    # model_id = request.args.get('modelId')
    # model_description = {
    #     "modelName": "H-JORTH SVM",
    #     "featureExtractor": "H-JORTH Desecriptor",
    #     'machineModel': "Support Vector Machine",
    #     "classes": [
    #         {"classLabel": "0", "className": "Normal Sinus Rhythm"},
    #         {"classLabel": "1", "className": "Atrial Firbrillation"},
    #         {"classLabel": "2", "className": "Congestive Heart Failure"},
    #     ]
    # }

    return "model"


def calculateActivity(epoch):
    activity = np.nanvar(epoch, axis=0)
    return activity


def calcMobility(epoch):
    mobility = np.divide(
        np.nanstd(np.diff(epoch, axis=0)),
        np.nanstd(epoch, axis=0))
    return mobility


def calcComplexity(epoch):
    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)),
        calcMobility(epoch))
    return complexity


if __name__ == '__main__':
    #app.run(host="127.0.0.1", debug=True, port=5000)
    print('run app now')
    app.run(host='0.0.0.0', port=8080, debug=True)
