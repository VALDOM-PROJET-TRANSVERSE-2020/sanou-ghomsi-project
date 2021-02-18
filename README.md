This project is aims to industrialize a Kaggle notebook from scratch.
A machine learning python notebook with training done with Tensorflow/Keras package is processed, 
the model is saved and used in an API that can be requested to perform ML predictions

## This project includes following steps:
#### 1. Find a notebook corresponding to the criteria requested on Kaggle
We have chosen the notebook in the following link:
https://www.kaggle.com/michalbrezk/x-ray-pneumonia-cnn-tensorflow-2-0-keras-94
 <br/>The corresponding dataset is in the following link:
 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 <br/> We save the trained model to be able to load it into the API

#### 2. Develop an API in Flask to serve the model
Using PyCharm and a virtual environment for development
Coding to load the model at API startup
Creating a route for inference
Use a WSGI HTTP web server (Gunicorn)
#### 3. Add Swagger documentation to the API
#### 4. Contain your API with Docker :
  Use docker-composite to define the complete stack
  
## Step to run the project

- Clone the repo
- Install Docker and Docker-compose component
- Build new image by typing this in terminal :
- On linux :`$ run_docker.sh`
- On Windows : `> run_docker_win.bat`
- To test the API, using Swagger : 
   - On a browser (Chrome or Firefox) open https://editor.swagger.io/
   - Click on *File*, then *Import url*
   - Insert this adress : `http://localhost:8000/static/openapi.json`
   - Play with the different routes.
  
