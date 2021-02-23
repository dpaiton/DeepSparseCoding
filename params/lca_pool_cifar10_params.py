import os
import types
import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams
from DeepSparseCoding.params.lca_cifar10_params import params as LcaParams
from DeepSparseCoding.utils.run_utils import compute_conv_output_shape


class shared_params(object):
    def __init__(self):
        self.model_type = 'ensemble'
        self.model_name = 'lca_pool_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.standardize_data = True
        self.batch_size = 25
        self.num_epochs = 5
        self.train_logs_per_epoch = 4
        self.allow_parent_grads = False


class lca_params(LcaParams):
    def set_params(self):
        super(lca_params, self).set_params()
        for key, value in shared_params().__dict__.items(): setattr(self, key, value)
        self.model_type = 'lca'
        self.layer_types = ['conv']
        self.weight_decay = 0.0
        self.weight_lr = 0.001
        self.renormalize_weights = True
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
        self.sparse_mult = 0.30
        self.num_latent = 512
        self.checkpoint_boot_log = '/mnt/qb/bethge/dpaiton/Projects/conv_lca_cifar10/logfiles/conv_lca_cifar10_v1.log'
        self.compute_helper_params()


class pool_params(BaseParams):
    def set_params(self):
        super(pool_params, self).set_params()
        for key, value in shared_params().__dict__.items():
          setattr(self, key, value)
        self.model_type = 'pool'
        self.layer_name = 'pool_1'
        self.weight_lr = 1e-3
        self.layer_type = 'conv'
        self.layer_channels = [512, 10]
        self.pool_ksize = 4
        self.pool_stride = 2
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.3] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.compute_helper_params()

    def compute_helper_params(self):
        super(pool_params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        lca_params_inst = lca_params()
        pool_params_inst = pool_params()
        if(pool_params_inst.layer_type == 'fc' and lca_params_inst.layer_type == 'conv'):
            lca_output_height = compute_conv_output_shape(
                32, # TODO: infer this? currently hardcoded CIFAR10 size
                lca_params_inst.kernel_size,
                lca_params_inst.stride,
                lca_params_inst.padding,
                dilation=1)
            lca_output_width = compute_conv_output_shape(
                32,
                lca_params_inst.kernel_size,
                lca_params_inst.stride,
                lca_params_inst.padding,
                dilation=1)
            lca_output_shape = [lca_params_inst.num_latent, lca_output_height, lca_output_width]
            pool_params_inst.layer_channels[0] = np.prod(lca_output_shape)
        self.ensemble_params = [lca_params_inst, pool_params_inst]
        for key, value in shared_params().__dict__.items():
            setattr(self, key, value)
