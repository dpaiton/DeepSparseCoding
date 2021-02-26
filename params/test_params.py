import os
import sys
import types
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.realpath(__file__))))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams
from DeepSparseCoding.params.lca_mnist_params import params as LcaParams
from DeepSparseCoding.params.mlp_mnist_params import params as MlpParams


class shared_params(object):
    def __init__(self):
        self.model_type = 'ensemble'
        self.model_name = 'test_ensemble'
        self.version = '0'
        self.dataset = 'synthetic'
        self.out_dir = os.path.join(*[ROOT_DIR, 'Projects', 'tests'])
        self.model_name = 'test'
        self.version = '0.0'
        self.dataset = 'synthetic'
        self.shuffle_data = True
        self.num_epochs = 2
        self.epoch_size = 30
        self.batch_size = 10
        self.data_edge_size = 8
        self.num_pixels = int(self.data_edge_size**2)
        self.dist_type = 'gaussian'
        self.num_classes = 10
        self.num_val_images = 0
        self.num_test_images = 0
        self.standardize_data = False
        self.rescale_data_to_one = False
        self.allow_parent_grads = False
        self.num_epochs = 3
        self.train_logs_per_epoch = 1


class base_params(BaseParams):
    def set_params(self):
        super(base_params, self).set_params()
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)


class lca_params(BaseParams):
    def set_params(self):
        super(lca_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'lca'
        self.weight_decay = 0.0
        self.weight_lr = 0.1
        self.layer_types = ['fc']
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.7] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.5
        self.renormalize_weights = True
        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = 'soft'
        self.sparse_mult = 0.25
        self.layer_channels = 128
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.step_size = self.dt / self.tau

# TODO: Add ability to test multiple param values
#class conv_lca_params(lca_params):
#    def set_params(self):
#        super(conv_lca_params, self).set_params()
#        self.layer_types = ['conv']
#        self.kernel_size = 8
#        self.stride = 2
#        self.padding = 0
#        self.optimizer.milestones = [frac * self.num_epochs
#            for frac in self.optimizer.lr_annealing_milestone_frac]
#        self.step_size = self.dt / self.tau
#        self.out_channels = self.layer_channels
#        self.in_channels = 1


class pooling_params(BaseParams):
    def set_params(self):
        super(pooling_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'pooling'
        self.layer_name = 'test_pool_1'
        self.weight_lr = 1e-3
        self.layer_types = ['conv']
        self.layer_channels = [128, 32]
        self.pool_ksize = 2
        self.pool_stride = 2 # non-overlapping
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


class mlp_params(BaseParams):
    def set_params(self):
        super(mlp_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'mlp'
        self.weight_lr = 1e-4
        self.weight_decay = 0.0
        self.layer_types = ['fc']
        self.layer_channels = [128, 10]
        self.activation_functions = ['identity']
        self.dropout_rate = [0.0] # probability of value being set to zero
        self.max_pool = [False]
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'adam'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.9
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


class ensemble_params(BaseParams):
    def set_params(self):
        super(ensemble_params, self).set_params()
        self.ensemble_params = [lca_params(), mlp_params()]
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)
