from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

model = load_model('./h_jorth_logistic_regression')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.DataFrame([[1154.21, 0.32, 1.60]], columns=[
        'activity', 'mobility', 'complexity'])
    predictions = predict_model(model, data=df)
    print(predictions)
    return 'hello world'


if __name__ == '__main__':
    app.run(debug=True)
