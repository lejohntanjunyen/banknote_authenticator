from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

@app.route("/")
def welcome():
    return "Welcome All!"

@app.route("/predict", methods=["Get"])
def predict_note_authentication():  
    """
    Let's Authenticate Bank Notes
    This is using docstrings for specifications
    ---
    parameters:
      - name: variance
        in : query
        type: number
        required: true
      - name: skewness
        in : query
        type: number
        required: true
      - name: curtosis
        in : query
        type: number
        required: true
      - name: entropy
        in : query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return "The predicted value is " + str(prediction)

@app.route("/predict_file", methods=["POST"])
def predict_note_file():  
    """
    Let's Authenticate Bank Notes
    This is using docstrings for specifications
    ---
    parameters:
      - name: file
        in : formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    test_data = pd.read_csv(request.files.get("file"))
    prediction = model.predict(test_data)
    return "The predicted value for the file is " + str(list(prediction))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)

