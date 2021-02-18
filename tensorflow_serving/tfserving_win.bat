echo Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

docker run -p 8600:8600 --name tfservingv1 --mount type=bind,source="C:\Users\ghomsik\Valdom-Projects\sanou-ghomsi-project\tensorflow_serving\tmp\model",target=/models/pnector -e MODEL_NAME=pnector -it tensorflow/serving