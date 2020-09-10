import os
import types

import numpy as np
import torch


class BaseParams(object):
    """
    all models
        batch_size [int] number of images in a training batch
        data_dir [str] location of dataset folders
        device [str] which device to run on
        dtype [torch dtype] dtype for network variables
        eps [float] small value to avoid division by zero
        fast_mnist [bool] if True, use the fastMNIST dataset,
            which loads faster but does not allow for torchvision transforms like flip and rotate
        lib_root_dir [str] system location of this library directory
        log_to_file [bool] if set, log to file, else log to stderr
        model_name [str] name for model (can be anything)
        model_type [str] type of model (must be among the list returned by utils/loaders.get_module_list())
        num_epochs [int] how many epochs to use for training
        num_pixels [int] total number of pixels in the input image
        out_dir [str] base directory for all model outputs
        optimizer [object] empty object with the following properties:
            optimizer.name [str] which optimization algorithm to use
                can be "sgd" (default) or "adam"
            optimizer.lr_annealing_milestone_frac [list of floats] fraction of num_epochs to anneal learning rate
            optimizer.lr_decay_rate [float] amount to anneal learning rate by
        rand_seed [int] seed to be given to np.random.RandomState
        rand_state [int] random state to be used for all random functions to allow for reproducible results
        renormalize_weights [bool] if set, l2 normalize weights after each update
        rescale_data_to_one [bool] if set, rescale input data to be between 0 and 1, per example
        version [str] model version for output
        shuffle_data [bool] if set, shuffle loader data before delivering batches
        standardize_data [bool] if set, z-score data to have mean=0 and standard deviation=1 using numpy operators
        train_logs_per_epoch [int or None] how often to send updates to the logfile
        workspace_dir [str] system directory that is the parent to the primary repository directory

    mlp
        activation_functions [list of str]  strings correspond to activation functions for layers.
            len must equal the len of layer_types
            strings must be one of those listed in modules/activations.activation_picker()
        dropout_rate [list of floats] specifies dropout probability of a value being set to zero or None per layer
            len must be equal to the len of layer_types
        layer_types [list of str] weight connectivity type, either "conv" or "fc"
            len must be equal to the len of layer_channels - 1
        layer_channels [list of int] number of outputs per layer, including the input layer

    lca
        dt [float] discrete global time constant for neuron dynamics
            lca update rule is multiplied by dt/tau
        num_latent [int] number of lca latent units
        num_steps [int] number of lca inference steps to take
        rectify_a [bool] if set, rectify the layer 1 neuron activity
        sparse_mult [float] multiplyer placed in front of the sparsity loss term
        tau [float] LCA time constant
            lca update rule (step_size) is multiplied by dt/tau
        thresh_type [str] specifying LCA threshold function; can be "hard" or "soft"

    lca; mlp
        weight_decay [float] multiplier to use on top of weight decay loss term
        weight_lr [float] learning rate for all weight updates
    """

    def __init__(self):
        self.set_params()
        self.compute_helper_params()

    def set_params(self):
        self.standardize_data = False
        self.rescale_data_to_one = False
        self.model_type = None
        self.log_to_file = True
        self.train_logs_per_epoch = None
        self.dtype = torch.float
        self.shuffle_data = True
        self.eps = 1e-12
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rand_seed = 123456789
        self.rand_state = np.random.RandomState(self.rand_seed)
        self.workspace_dir = os.path.join(os.path.expanduser('~'), 'Work')
        self.lib_root_dir = os.path.join(self.workspace_dir, 'DeepSparseCoding')
        self.data_dir = os.path.join(self.workspace_dir, 'Datasets')
        self.out_dir = os.path.join(self.workspace_dir, 'Torch_projects')

    def compute_helper_params(self):
        pass

class params(BaseParams):
    pass
