# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving
#cp -r model/ /tmp/
docker run -p 8600:8600 --name tfservingv1 --mount type=bind,source=/tmp/model,target=/models/pnector -e MODEL_NAME=pnector -it tensorflow/serving