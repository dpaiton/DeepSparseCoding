import os
import sys
import unittest
import types

import numpy as np
from torchvision import datasets

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.dataset_utils as dataset_utils

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(ROOT_DIR, 'Datasets')

    #def test_mnist(self):
    #    try: # only run the test if the dataset is already downloaded
    #        mnist = datasets.MNIST(root=self.data_dir, train=True, download=False)
    #    except:
    #        return 0
    #    standardize_data_list = [True, False]
    #    for standardize_data in standardize_data_list:
    #        params = types.SimpleNamespace()
    #        params.standardize_data = standardize_data
    #        if(params.standardize_data):
    #            params.eps = 1e-8
    #        params.data_dir = self.data_dir
    #        params.dataset = 'mnist'
    #        params.shuffle_data = True
    #        params.batch_size = 10000
    #        train_loader, val_loader, test_loader, params = dataset_utils.load_dataset(params)
    #        assert len(train_loader.dataset) == params.epoch_size
    #        (data, target) = next(iter(train_loader))
    #        assert data.numpy().shape == (params.batch_size, 28, 28, 1) 

    def test_synthetic(self):
        epoch_size_list = [20, 50]
        data_edge_size_list = [3, 8]
        dist_type_list = ['gaussian', 'laplacian']
        num_classes_list = [None, 10]
        rand_state_list = [np.random.RandomState(12345)]
        for epoch_size in epoch_size_list:
            for data_edge_size in data_edge_size_list:
                for dist_type in dist_type_list:
                    for num_classes in num_classes_list:
                        for rand_state in rand_state_list:
                            params = types.SimpleNamespace()
                            params.dataset = 'synthetic'
                            params.shuffle_data = True
                            params.epoch_size = epoch_size
                            params.batch_size = 10
                            params.data_edge_size = data_edge_size
                            params.dist_type = dist_type
                            params.num_classes = num_classes
                            params.rand_state = rand_state
                            train_loader, val_loader, test_loader, params = dataset_utils.load_dataset(params)
                            assert len(train_loader.dataset) == epoch_size
                            for batch_idx, (data, target) in enumerate(train_loader):
                               assert data.numpy().shape == (params.batch_size, params.data_edge_size, params.data_edge_size, 1) 
                            assert batch_idx + 1 == epoch_size // params.batch_size
