from params.ica_params import params as ica_params

import os 
import numpy as np 


class params(ica_params):
    def __init__(self):
        super(params, self).__init__()
        self.model_type = "ica_subspace"
        self.model_name = "ica_subspace"
        self.version = "0.0"
        self.data_type = "vanHateren"

        self.batch_size = 1


        self.num_groups = 16
        self.group_sizes = None

        self.center_data = False
        self.whiten_data = False
        self.lpf_data = False
        self.whiten_method = "PCA"

        self.optimizer = "adam"

        self.schedule = [
            {"weights": None, 
             "num_batches": int(1e5),
             "weight_lr": 0.001, 
             "decay_steps": 0,
             "decay_rate": 0.0,
             "staircase": False}
        ]

