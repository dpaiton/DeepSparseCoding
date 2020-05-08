import os
import sys
import unittest


ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.loaders as loaders


class TestModels(unittest.TestCase):
    def test_model_loading(self):
        dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
        model_list = loaders.get_model_list(dsc_dir)
        for model_type in model_list:
            model_type = ''.join(model_type.split("_")[:-1]) # remove '_model' at the end
            model = loaders.load_model(model_type)

    def test_gradients(self):
        dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
        model_list = loaders.get_model_list(dsc_dir)
        for model_type in model_list:
            model_type = ''.join(model_type.split("_")[:-1]) # remove '_model' at the end
            model = loaders.load_model(model_type)

