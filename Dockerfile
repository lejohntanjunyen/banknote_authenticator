FROM continuumio/anaconda3:4.4.0
COPY D:\PROJECTS\banknote
EXPOSE 5000
WORKDIR D:\PROJECTS\banknote
RUN pip install -r requirements.txt
CMD python flask_api.py