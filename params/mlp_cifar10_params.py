import os
import types

import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = 'mlp'
        self.model_name = 'mlp_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.standardize_data = True
        self.rescale_data_to_one = False
        self.center_data = False
        self.num_validation = 1000
        self.batch_size = 50
        self.num_epochs = 500
        self.weight_decay = 3e-6
        self.weight_lr = 2e-3
        self.layer_types = ['conv', 'fc']
        self.layer_channels = [3, 512, 10]
        self.kernel_sizes = [8, None]
        self.strides = [2, None]
        self.activation_functions = ['lrelu', 'identity']
        self.dropout_rate = [0.5, 0.0] # probability of value being set to zero
        self.max_pool = [True, False]
        self.pool_ksizes = [5, None]
        self.pool_strides = [4, None]
        self.train_logs_per_epoch = 4
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'adam'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.1

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
