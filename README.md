This project is aims to industrialize a Kaggle notebook from scratch.
A machine learning python notebook with training done with Tensorflow/Keras package is processed, 
the model is saved and used in an API that can be requested to perform ML predictions

This project includes following steps:
1. Find a notebook corresponding to the criteria requested on Kaggle
We have chosen the notebook (Pneumonia detection based on Chest X-Ray images on imbalanced data using DL) in the following link:
https://www.kaggle.com/michalbrezk/x-ray-pneumonia-cnn-tensorflow-2-0-keras-94
 <br/>The corresponding dataset is in the following link:
 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 <br/> We save the trained model to be able to load it into the API

2. Develop an API in Flask to serve the model
Configure your IDE (e.g. PyCharm) to use a virtual environment
Coding to load the model at API startup
Creating a route for inference
Use a WSGI HTTP web server (e.g. Gunicorn)
3. Add Swagger documentation to the API
Contain your API with Docker :
  Use docker-composite to define the complete stack
