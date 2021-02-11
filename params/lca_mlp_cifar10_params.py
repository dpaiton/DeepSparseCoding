import os
import types
import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams
from DeepSparseCoding.params.lca_mnist_params import params as LcaParams
from DeepSparseCoding.params.mlp_mnist_params import params as MlpParams


class shared_params(object):
    def __init__(self):
        self.model_type = 'ensemble'
        self.model_name = 'lca_mlp_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.standardize_data = True
        self.batch_size = 25
        self.num_epochs = 500
        self.train_logs_per_epoch = 4
        self.allow_parent_grads = False


class lca_params(LcaParams):
    def set_params(self):
        super(lca_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'lca'
        self.weight_decay = 0.0
        self.weight_lr = 0.001
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.renormalize_weights = True
        self.dt = 0.001
        self.tau = 0.2
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = 'hard'
        self.sparse_mult = 0.30
        self.num_latent = 512
        self.checkpoint_boot_log = ''
        self.compute_helper_params()


class mlp_params(MlpParams):
    def set_params(self):
        super(mlp_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'mlp'
        self.weight_lr = 2e-3
        self.weight_decay = 1e-6
        self.layer_types = ['fc']
        self.layer_channels = [512, 10]
        self.activation_functions = ['identity']
        self.dropout_rate = [0.0] # probability of value being set to zero
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'adam'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.compute_helper_params()


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.ensemble_params = [lca_params(), mlp_params()]
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)
