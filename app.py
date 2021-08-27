#from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
#import flasgger
#from flasgger import Swagger
import streamlit as st

#app = Flask(__name__)
#Swagger(app)

pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

#@app.route("/")
def welcome():
    return "Welcome All!"

#@app.route("/predict", methods=["Get"])
def predict_note_authentication(variance, skewness, curtosis, entropy):  
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
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is " + str(prediction)

def main():
  st.title("Bank Note Authenticator")
  html_temp = """
  <div style="background-color:tomato;padding:10px">
  <h2 style="color:white;text-align:center;">Streamlit Bank Note Authenticator ML App </h2>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)
  variance = st.text_input("variance")
  skewness = st.text_input("skewness")
  curtosis = st.text_input("curtosis")
  entropy = st.text_input("entropy")
  result=""
  if st.button("Predict"):
    result = predict_note_authentication(variance, skewness, curtosis, entropy)
  st.success('The Output is {}'.format(result))
  if st.button("About"):
    st.text("[1] means the bank note is authentic, [0] means the bank note is not authentic")
    st.text("Examples for output 0:")
    st.text("[1, 4, -1, -0.5]")
    st.text("[5, 8, -4, -3]")
    st.text("Examples for output 1:")
    st.text("[-0.1, -5, -2, 0.1]")
    st.text("[-4, 8, 3, -7]")


if __name__=="__main__":
    main()
