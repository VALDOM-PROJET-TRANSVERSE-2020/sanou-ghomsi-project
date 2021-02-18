FROM python:3.8.7


WORKDIR usr/src/flask_app


COPY . .
# OR
#COPY /requirements.txt .
#COPY __init__.py /flask_app
#COPY api.py /flask_app
#COPY pneu_detector.py /flask_app
#COPY templates /flask_app/templates
#COPY static /flask_app/static
#COPY uploads /flask_app/uploads
#COPY ml_model /flask_app/ml_model

RUN pip install -r requirements.txt --ignore-installed


EXPOSE 8000

#Starting the python application
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "wsgi:app"]
#CMD ["python", "./app/api.py"]