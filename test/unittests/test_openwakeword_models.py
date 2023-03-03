# imports
import unittest
import numpy as np
import openwakeword

class TestOpenWakeWord_plugin(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = openwakeword.Model()  # load default models

    def test_model_prediction(self):
        for i in range(10):
            prediction = self.model.predict(np.zeros(1280).astype(np.int16))
        self.assertNotEqual(0.0, prediction[list(self.model.models.keys())[0]])
