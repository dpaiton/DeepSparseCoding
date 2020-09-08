import os
import types

import numpy as np
import torch

from DeepSparseCoding.params.base_params import BaseParams


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = 'lca'
        self.model_name = 'lca_dsprites'
        self.version = '0'
        self.dataset = 'dsprites'
        self.standardize_data = False
        self.num_pixels = 4096
        self.batch_size = 1024
        self.num_epochs = 1000
        self.weight_decay = 0.
        self.weight_lr = 0.1
        self.train_logs_per_epoch = 6
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.7] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.5
        self.renormalize_weights = True
        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = False
        self.thresh_type = 'soft'
        self.sparse_mult = 0.25
        self.num_latent = int(self.num_pixels*1.5)
        self.compute_helper_params()

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.step_size = self.dt / self.tau
