import os
import types

import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams
from DeepSparseCoding.params.lca_cifar10_params import params as LcaParams
from DeepSparseCoding.params.mlp_mnist_params import params as MlpParams
from DeepSparseCoding.utils.run_utils import compute_conv_output_shape


class shared_params(object):
    def __init__(self):
        self.model_type = 'ensemble'
        self.model_name = 'lca_pool_lca_pool_mlp_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.standardize_data = True
        self.batch_size = 25
        self.num_epochs = 150
        self.train_logs_per_epoch = 4
        self.allow_parent_grads = False


class lca_1_params(LcaParams):
    def set_params(self):
        super(lca_1_params, self).set_params()
        for key, value in shared_params().__dict__.items(): setattr(self, key, value)
        self.model_type = 'lca'
        self.layer_name = 'lca_1'
        self.layer_types = ['conv']
        self.weight_decay = 0.0
        self.weight_lr = 0#1e-3
        self.renormalize_weights = True
        self.layer_channels = 128
        self.kernel_size = 8
        self.stride = 2
        self.padding = 0
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.dt = 0.001
        self.tau = 0.2
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = 'hard'
        self.sparse_mult = 0.35#0.30
        self.checkpoint_boot_log = '/mnt/qb/bethge/dpaiton/Projects/lca_pool_lca_pool_cifar10/logfiles/lca_pool_lca_pool_cifar10_v0.log'
        self.compute_helper_params()


class pooling_1_params(BaseParams):
    def set_params(self):
        super(pooling_1_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'pooling'
        self.layer_name = 'pool_1'
        self.weight_lr = 0#1e-3
        self.layer_types = ['conv']
        self.layer_channels = [128, 32]
        self.pool_ksize = 2
        self.pool_stride = 2 # non-overlapping
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.checkpoint_boot_log = '/mnt/qb/bethge/dpaiton/Projects/lca_pool_lca_pool_cifar10/logfiles/lca_pool_lca_pool_cifar10_v0.log'
        self.compute_helper_params()

    def compute_helper_params(self):
        super(pooling_1_params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


class lca_2_params(LcaParams):
    def set_params(self):
        super(lca_2_params, self).set_params()
        for key, value in shared_params().__dict__.items(): setattr(self, key, value)
        for key, value in lca_1_params().__dict__.items(): setattr(self, key, value)
        self.layer_name = 'lca_2'
        self.weight_lr = 0#1e-3
        self.layer_channels = 256
        self.kernel_size = 6
        self.stride = 1
        self.padding = 0
        self.sparse_mult = 0.20
        self.checkpoint_boot_log = '/mnt/qb/bethge/dpaiton/Projects/lca_pool_lca_pool_cifar10/logfiles/lca_pool_lca_pool_cifar10_v0.log'
        self.compute_helper_params()

class pooling_2_params(BaseParams):
    def set_params(self):
        super(pooling_2_params, self).set_params()
        for key, value in shared_params().__dict__.items(): setattr(self, key, value)
        for key, value in pooling_1_params().__dict__.items(): setattr(self, key, value)
        self.layer_name = 'pool_2'
        self.weight_lr = 0#1e-3
        self.layer_types = ['fc']
        self.layer_channels = [None, 64]
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.checkpoint_boot_log = '/mnt/qb/bethge/dpaiton/Projects/lca_pool_lca_pool_cifar10/logfiles/lca_pool_lca_pool_cifar10_v0.log'
        self.compute_helper_params()

    def compute_helper_params(self):
        super(pooling_2_params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


class mlp_params(MlpParams):
    def set_params(self):
        super(mlp_params, self).set_params()
        for key, value in shared_params().__dict__.items(): setattr(self, key, value)
        self.model_type = 'mlp'
        self.layer_name = 'classifier'
        self.weight_lr = 2e-3
        self.weight_decay = 1e-6
        self.layer_types = ['fc']
        self.layer_channels = [64, 10]
        self.activation_functions = ['identity']
        self.dropout_rate = [0.0] # probability of value being set to zero
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.compute_helper_params()


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        lca_1_params_inst = lca_1_params()
        pooling_1_params_inst = pooling_1_params()
        lca_2_params_inst = lca_2_params()
        pooling_2_params_inst = pooling_2_params()
        mlp_params_inst = mlp_params()
        lca_1_output_height = compute_conv_output_shape(
            32,
            lca_1_params_inst.kernel_size,
            lca_1_params_inst.stride,
            lca_1_params_inst.padding,
            dilation=1)
        lca_1_output_width = compute_conv_output_shape(
            32,
            lca_1_params_inst.kernel_size,
            lca_1_params_inst.stride,
            lca_1_params_inst.padding,
            dilation=1)
        pooling_1_output_height = compute_conv_output_shape(
            lca_1_output_height,
            pooling_1_params_inst.pool_ksize,
            pooling_1_params_inst.pool_stride,
            padding=0,
            dilation=1)
        pooling_1_output_width = compute_conv_output_shape(
            lca_1_output_width,
            pooling_1_params_inst.pool_ksize,
            pooling_1_params_inst.pool_stride,
            padding=0,
            dilation=1)
        lca_2_params_inst.data_shape = [
            int(pooling_1_params_inst.layer_channels[-1]),
            int(pooling_1_output_height),
            int(pooling_1_output_width)]
        lca_2_output_height = compute_conv_output_shape(
            pooling_1_output_height,
            lca_2_params_inst.kernel_size,
            lca_2_params_inst.stride,
            lca_2_params_inst.padding,
            dilation=1)
        lca_2_output_width = compute_conv_output_shape(
            pooling_1_output_width,
            lca_2_params_inst.kernel_size,
            lca_2_params_inst.stride,
            lca_2_params_inst.padding,
            dilation=1)
        lca_2_flat_dim = lca_2_params_inst.layer_channels*lca_2_output_height*lca_2_output_width
        pooling_2_params_inst.layer_channels[0] = lca_2_flat_dim
        self.ensemble_params = [
            lca_1_params_inst,
            pooling_1_params_inst,
            lca_2_params_inst,
            pooling_2_params_inst,
            mlp_params_inst
        ]
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)
