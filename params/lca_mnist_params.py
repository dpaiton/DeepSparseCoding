import types

from DeepSparseCoding.params.base_params import BaseParams


CONV = True


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = 'lca'
        self.version = '0'
        self.dataset = 'mnist'
        self.fast_mnist = True
        self.standardize_data = False
        self.num_pixels = 784
        self.dt = 0.001
        self.tau = 0.03
        self.num_steps = 75
        self.rectify_a = True
        self.thresh_type = 'soft'
        self.sparse_mult = 0.25
        self.renormalize_weights = True
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.num_epochs = 1000
        self.weight_decay = 0.0
        self.train_logs_per_epoch = 6
        if CONV:
            self.layer_types = ['conv']
            self.model_name = 'conv_lca_mnist'
            self.rescale_data_to_one = True
            self.batch_size = 50
            self.weight_lr = 0.001
            self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
            self.optimizer.lr_decay_rate = 0.8
            self.kernel_size = 8
            self.stride = 2
            self.padding = 0
            self.num_latent = 128
        else:
            self.layer_types = ['fc']
            self.model_type = 'lca'
            self.model_name = 'lca_768_mnist'
            self.rescale_data_to_one = False
            self.batch_size = 100
            self.weight_lr = 0.1
            self.optimizer.lr_annealing_milestone_frac = [0.7] # fraction of num_epochs
            self.optimizer.lr_decay_rate = 0.5
            self.num_latent = 768 #self.num_pixels * 4
        self.compute_helper_params()

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.step_size = self.dt / self.tau
