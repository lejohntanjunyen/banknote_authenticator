Mini Project on Bank Note Authenticator
- built a machine learning model to predict whether the bank note is authentic or not
- deployed the model using Streamlit into Heroku
- https://banknote-authenticator-1006.herokuapp.com/

Files:
BankNote_Authentication.csv
- bank note data set

TestFile.csv
- test data set

BankNoteAuthenticationModel.ipynb
- RandomForestClassifier Model to predict authenticity of Bank Note

model.pkl
- pickled out RandomForestClassifier Model

prediction.py
- a simple python file to deploy model into web application using flask

flask_api.py 
- similar to prediction.py but included flasgger  

app.py 
- python file to deploy model using Streamlit

requirements.txt
- a lsit of required packages and its version

setup.sh
- shell file with shell command
- Note: do not open this file in your computer, will create a config.toml file inside the .streamlit file and running streamlit will pose problems

Procfile
- specifies the shell file to run app on Heroku


