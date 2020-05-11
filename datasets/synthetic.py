import os
import sys

import numpy as np
from scipy.stats import norm
from PIL import Image
import torch
import torchvision

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.data_processing as dp

class SyntheticImages(torchvision.datasets.vision.VisionDataset):
    """Synthetic dataset of square images with pixel values drawn from a specified distribution
    Inputs:
        epoch_size [int] Number of datapoints in the dataset
        data_edge_size [int] Length of the edge of a square datapoint.
        dist_type [str] one of {'gaussian', 'laplacian', 'hierarchical_sparse'}
        rand_state [np.random.RandomState()] a numpy random state to generate data from
        num_classes [int, optional] number of classes for random supervised labels
        transform [callable, optional] A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, epoch_size, data_edge_size, dist_type, rand_state, num_classes=None,
        transform=None, target_transform=None):
        root = './' # no need for a root directory because the data is never on disc
        if(target_transform):
            assert num_classes is not None, (
                'Num classes must be specified if target_transform is not None.')
        super(SyntheticImages, self).__init__(root, transform=transform,
            target_transform=target_transform) # transforms get set to member variables
        self.data = torch.tensor(
            self.generate_synthetic_data(epoch_size, data_edge_size, dist_type, rand_state))
        if(num_classes):
            self.targets = self.generate_labels(epoch_size, num_classes, rand_state)
        else:
            self.targets = -1 * torch.ones(len(self.data))

    def __getitem__(self, index):
        """
        Inputs:
            index (int): Index
        Outputs:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.squeeze(img.numpy()), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def generate_synthetic_data(self, epoch_size, data_edge_size, dist_type, rand_state):
        """
        Function for generating synthetic data of shape [epoch_size, num_edge, num_edge]
        Inputs:
            dist_type [str] one of {'gaussian', 'laplacian', 'hierarchical_sparse'},
                otherwise returns zeros
            epoch_size [int] number of datapoints in an epoch
            data_edge_size [int] size of the edge of the square synthetic image
        """
        data_shape = (epoch_size, data_edge_size, data_edge_size, 1)
        if dist_type == 'gaussian':
            data = rand_state.normal(loc=0.0, scale=1.0, size=data_shape)
        elif dist_type == 'laplacian':
            data = rand_state.laplace(loc=0.0, scale=1.0, size=data_shape)
        elif dist_type == 'hierarchical_sparse': # courtisy of Sophia Sanborn github.com/sophiaas
            # Number of second-level basis functions
            num_l2_functions = 10
            # First-level identity basis
            l1_basis = np.identity(data_edge_size**2)
            # Second-level Gaussian basis
            sig = data_edge_size // 2
            # l2_basis will have shape = [num_edge_pixels, num_l2_functions]
            l2_basis = np.asarray([
                norm.pdf(np.arange(0, data_edge_size), rand_state.randint(0, data_edge_size), sig)
                for i in range(num_l2_functions)], dtype=np.float32).T
            l2_basis /= (np.max(l2_basis, axis=0)[None, :])
            # Laplacian prior over l2_basis
            v = rand_state.laplace(0, 1, (num_l2_functions, epoch_size))
            lam = np.exp(np.dot(l2_basis, v)) # shape = [num_edge_pixels, epoch_size]
            u = np.sqrt(lam) * rand_state.normal(0, 1, (data_edge_size**2, epoch_size))
            # Generate data
            data = np.reshape(np.dot(l1_basis, u), data_shape)
        else:
            data = np.zeros(data_shape)
        return data

    def generate_labels(self, epoch_size, num_classes, rand_state):
        #Generate classes from 2 avaliable classes
        labels = torch.tensor(rand_state.randint(num_classes, size=epoch_size))
        labels = dp.dense_to_one_hot(labels, num_classes)
        return labels
