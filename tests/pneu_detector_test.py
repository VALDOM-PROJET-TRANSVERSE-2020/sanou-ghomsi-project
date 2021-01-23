import unittest
from flask_api.pneu_detector import  *

pneumonia_model_path = '../ml_model/finalModel/pneu_detect_cnn_model.h5'
image_path= "../dataprep/chest_xray/test/"
model= PneumoniaDetector(pneumonia_model_path)

class PredictionsModelTestCase(unittest.TestCase):
    def test_data_process(self):
        pass


    def test_check_pneumonia(self):
        prediction_dict, filename= model.check_pneumonia(image_path+"IM-0001-0001.jpeg")
        expected_filename= "pred= " + prediction_dict["pred"] + ";confidence= " + prediction_dict["proba"]
        self.assertEqual(prediction_dict["pred"], "Normal Chest")
        self.assertEqual(prediction_dict["proba"]>0.5, True)
        self.assertEqual(filename,expected_filename)

    def test_normal_chest(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
