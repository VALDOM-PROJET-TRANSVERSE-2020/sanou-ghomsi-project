from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf
from tensorflow.keras.models import load_model

tf.compat.v1.disable_eager_execution()


def main(_):
  model_path= "../flask_app/ml_model/finalModel/pneu_detect_cnn_model.h5"
  model = load_model(model_path)
  export_path_base = "./tmp/model"
  export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(1)))
  print('Exporting trained model to', export_path)

  tf.compat.v1.keras.experimental.export_saved_model(model, export_path)

  # Build the signature_def_map.
  print('Done exporting!')

if __name__ == '__main__':
  tf.compat.v1.app.run()