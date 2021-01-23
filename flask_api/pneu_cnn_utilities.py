import glob
import random as rn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras



class ImageProcessingModel:

    
  
    def process_data(self,img_path):
        """ 
        process_data - load image, resize it, convert to grayscale,
        normalize and reshape to dimension required for tensorflow
        
        """ 
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (196,196))
        img = tf.cast(img / 255., tf.float32)
        img = tf.reshape(img, (1,196,196,1))
        img = np.array(img)
        
        return img
    
    def model_predict(self,img_proc, model_path):
        """
        load model and predict
        
        """
        model = keras.models.load_model(model_path)
        y_pred = model.predict(img_proc, batch_size=4)
        y_pred = np.argmax(y_pred, axis=1)
        
        if y_pred == 1:
            return "PNEUMONIA"
        elif y_pred == 0:
            return "NORMAL"
        else:
            return "Cannot predict"
        