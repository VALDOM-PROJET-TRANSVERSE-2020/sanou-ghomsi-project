from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

class PneumoniaDetector(object):

    def __init__(self, model_path):
        self.prd_model = load_model(model_path)


    @staticmethod
    def __process_data(img_path):
        """
        process_data - load image, resize it, convert to grayscale,
        normalize and reshape to dimension required for tensorflow
        """
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (196, 196))
        img = tf.cast(img / 255., tf.float32)
        img = tf.reshape(img, (1, 196, 196, 1))
        img = np.array(img)
        return img

    def predict(self, image_file):
        img_proc = self.__class__.__process_data(image_file)
        pred_probas = self.prd_model.predict(img_proc)
        pred_index = np.argmax(pred_probas, axis=1)
        pred_confidence = pred_probas[0][pred_index]
        return pred_index, np.round(pred_confidence, 2)

    def check_pneumonia(self, image_file):
        error_message = "Failed to predict the class:  "
        try:
            has_pneumonia, pred_confidence = self.predict(image_file)
            if has_pneumonia:
                prediction = "Pneumonia detected"
            else:
                prediction = "Normal Chest"
            prediction_dict = {'pred': prediction, 'proba': pred_confidence[0]}
        except Exception as ex:
            prediction_dict = {'pred': error_message + str(ex), 'proba': "None"}
        return prediction_dict
