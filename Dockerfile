FROM python:3.8.7


RUN mkdir ./app
RUN mkdir ./ml_model

COPY ./env/requirements.txt /app
RUN pip install -r requirements.txt --ignore-installed
#RUN pip install gunicorn
COPY __init__.py /app
COPY ./flask_api/api.py /app
COPY ./flask_api/pneu_detector.py /app
COPY ./flask_api/templates /app/templates
COPY ./flask_api/static /app/static
COPY ./flask_api/uploads /app/uploads
COPY ./ml_model/finalModel /ml_model/finalModel
#COPY . .
WORKDIR /app
EXPOSE 5005

#Starting the python application
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
CMD ["python", "./app/api.py"]