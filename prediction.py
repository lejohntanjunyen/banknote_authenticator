from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

@app.route("/")
def welcome():
    return "Welcome All!"

@app.route("/predict")
def predict_note_authentication():  
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is" + str(prediction)

@app.route("/predict_file", methods=["POST"])
def predict_note_file():  
    test_data = pd.read_csv(request.files.get("file"))
    prediction = model.predict(test_data)
    return "The predicted values for the csv is" + str(list(prediction))

if __name__=="__main__":
    app.run()