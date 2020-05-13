import os
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.data_processing as dp
import DeepSparseCoding.datasets.synthetic as synthetic



def load_dataset(params):
    new_params = {}
    if(params.dataset.lower() == 'mnist'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=True, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=False, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        new_params["epoch_size"] = len(train_loader.dataset)
    elif(params.dataset.lower() == 'synthetic'):
        preprocessing_pipeline = [transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        train_loader = torch.utils.data.DataLoader(
            synthetic.SyntheticImages(params.epoch_size, params.data_edge_size, params.dist_type,
            params.rand_state, params.num_classes,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = None
        new_params["num_pixels"] = params.data_edge_size**2
    else:
        assert False, (f'Supported datasets are ["mnist"], not {dataset_name}')
    new_params["data_shape"] = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, new_params)
