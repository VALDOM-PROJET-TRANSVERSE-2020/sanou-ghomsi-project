import logging

import requests
import json

def try_tf_serving(instances, url):
    data = json.dumps(
        {"signature_name": "serving_default", "instances": instances}
    )
    headers = {"content-type": "application/json"}


    response = requests.post(url, timeout = 10, data=data, headers=headers)


    predictions = json.loads(response.file)["predictions"]

    error_message = "Failed to predict the class:  "
    try:
        if predictions[0][0]>0.5:
            prediction = "Normal Chest"
            logging.info("Prediction :"+ prediction)
        else:
            prediction = "Pneumonia"
            logging.info("Prediction :"+ prediction)
    except Exception as ex:
        logging.error("Prediction :" + error_message + str(ex))
    return prediction



url = "http://localhost:8501/v1/models/pnector:predict"

if __name__ == "__main__":
    image_path= "../dataprep/chest_xray/test/"
    normal_image_file = image_path+"NORMAL/IM-0001-0001.jpeg"
    infected_image_file = image_path+"PNEUMONIA/person1_virus_6.jpeg"
    try_tf_serving(normal_image_file, url)
    try_tf_serving(infected_image_file, url)