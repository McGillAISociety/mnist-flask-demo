import unittest
import os
from PIL import Image
import numpy as np

from model_flax_vit import ViTPipeline, load_saved_model

TEST_VIT_MODEL = os.environ.get("TEST_VIT_MODEL", "google/vit-base-patch16-224")


class ViTPipelineTest(unittest.TestCase):
    def setUp(self):
        self.example_vit_pipeline = ViTPipeline(TEST_VIT_MODEL)

    def test_pipeline_predict(self):
        example_image_array = np.zeros((256, 256))
        example_image = Image.fromarray(example_image_array)
        example_prediction_output = self.example_vit_pipeline.predict(example_image)

        self.assertIsInstance(example_prediction_output[0], int)
        self.assertIsInstance(example_prediction_output[1], str)

    def test_load_pipeline_function(self):
        loaded_pipeline = load_saved_model(TEST_VIT_MODEL)
        example_image_array = np.ones((256, 256))
        example_image = Image.fromarray(example_image_array)
        example_prediction_output = loaded_pipeline.predict(example_image)

        self.assertIsInstance(example_prediction_output[0], int)
        self.assertIsInstance(example_prediction_output[1], str)
