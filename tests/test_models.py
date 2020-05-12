import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.loaders as loaders
import DeepSparseCoding.utils.dataset_utils as datasets
#import DeepSparseCoding.utils.file_utils as file_utils


class TestModels(unittest.TestCase):
    def setUp(self):
        self.dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
        self.model_list = loaders.get_model_list(self.dsc_dir)
        self.extra_params = {}
        self.extra_params['out_dir'] = os.path.join(*[ROOT_DIR, 'Projects', 'tests'])
        self.extra_params['model_name'] = 'test'
        self.extra_params['version'] = '0.0'
        self.extra_params['dataset'] = 'synthetic'
        self.extra_params['shuffle_data'] = True
        self.extra_params['num_epochs'] = 2
        self.extra_params['epoch_size'] = 30
        self.extra_params['batch_size'] = 10
        self.extra_params['data_edge_size'] = 8
        self.extra_params['dist_type'] = 'gaussian'
        self.extra_params['num_classes'] = 10
        self.extra_params['num_val_images'] = 0
        self.extra_params['num_test_images'] = 0


    def test_model_loading(self):
        for model_type in self.model_list:
            model_type = ''.join(model_type.split("_")[:-1]) # remove '_model' at the end
            model = loaders.load_model(model_type)
            test_params_file = os.path.join(*[self.dsc_dir, 'params', 'test_params.py'])
            params = loaders.load_params(test_params_file, key=model_type+'_params')
            for key, value in self.extra_params.items():
                setattr(params, key, value)
            train_loader, val_loader, test_loader, params = datasets.load_dataset(params)
            model.setup(params)


    #def test_gradients(self):
    #    for model_type in self.model_list:
    #        model_type = ''.join(model_type.split("_")[:-1]) # remove '_model' at the end
    #        model = loaders.load_model(model_type)
