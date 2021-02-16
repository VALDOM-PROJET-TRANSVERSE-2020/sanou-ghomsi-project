FROM python:3.8.7


RUN mkdir ./app
RUN mkdir ./ml_model

COPY ../env/requirements.txt /app
RUN pip install -r requirements.txt --ignore-installed
#RUN pip install gunicorn
COPY __init__.py /app
COPY api.py /app
COPY pneu_detector.py /app
COPY templates /app/templates
COPY static /app/static
COPY uploads /app/uploads
COPY ../ml_model/finalModel /ml_model
#COPY . .
WORKDIR /app
EXPOSE 5005

#Starting the python application
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
CMD ["python", "./app/api.py"]