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
        self.model_name = 'lca_768_mlp_mnist'
        self.version = '0'
        self.dataset = 'mnist'
        self.fast_mnist = True
        self.standardize_data = False
        self.num_pixels = 28*28*1
        self.batch_size = 100
        self.num_epochs = 1200
        self.train_logs_per_epoch = 4
        self.allow_parent_grads = False


class lca_params(LcaParams):
    def set_params(self):
        super(lca_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'lca'
        self.layer_name = 'lca'
        self.weight_decay = 0.0
        self.weight_lr = 0.1
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.7] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.5
        self.renormalize_weights = True
        self.layer_channels = [1, 768]
        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = 'soft'
        self.sparse_mult = 0.25
        self.checkpoint_boot_log = ''
        self.compute_helper_params()


class mlp_params(MlpParams):
    def set_params(self):
        super(mlp_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'mlp'
        self.layer_name = 'classifier'
        self.weight_lr = 1e-4
        self.weight_decay = 0.0
        self.layer_types = ['fc']
        self.layer_channels = [768, 10]
        self.activation_functions = ['identity']
        self.dropout_rate = [0.0] # probability of value being set to zero
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'adam'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.9
        self.compute_helper_params()


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.ensemble_params = [lca_params(), mlp_params()]
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)
