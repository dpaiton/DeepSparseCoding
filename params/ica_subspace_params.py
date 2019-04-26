from params.ica_params import params as ica_params

import os 
import numpy as np 


class params(ica_params):
    def __init__(self):
        super(params, self).__init__()

        # model config
        self.model_type = "ica_subspace"
        self.model_name = "ica_subspace"
        self.version = "0.1"

        # data config
        self.data_type = "vanHateren"
        self.batch_size = 50
        self.norm_data = False
        

        # hyperparams
        self.num_groups = 64
        self.group_sizes = [4 for _ in range(self.num_groups)]

        # data preprocessing
        self.center_data = True 
        self.whiten_data = True 
        self.lpf_data = True
        self.whiten_method = "ZCA"

        # model output intervals
        self.gen_plot_int = 2e3
        self.cp_int = 5e3
        self.log_int = 100

        # directories
        self.display_dir = os.path.expanduser("~")+"/Work/Projects/"+self.model_name+"/vis/"

        self.schedule = [
            {"weights": None, 
             "num_batches": 10,# int(1e4),
             "weight_lr": float(1), 
             "decay_steps": 1,
             "decay_rate": 1.0,
             "staircase": False}
        ]


