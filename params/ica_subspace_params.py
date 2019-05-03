from params.ica_params import params as ica_params

import os 
import numpy as np 


class params(ica_params):
    def __init__(self):
        super(params, self).__init__()

        # model config
        self.model_type = "ica_subspace"
        self.model_name = "ica_subspace"
        self.version = "0.0"

        # data config
        self.data_type = "vanHateren"
        self.batch_size = 50
        self.norm_data = False

        # hyperparams
        self.num_groups = 64
        self.alpha = 1
        self.beta = 0

        # data preprocessing
        self.center_data = True 
        self.whiten_data = True 
        self.lpf_data = True
        self.whiten_method = "ZCA"
        self.whiten_batch_size = 10

        # model output intervals
        self.gen_plot_int = 2e3
        self.cp_int = 5e3
        self.log_int = 100

        # directories
        self.display_dir = os.path.expanduser("~")+"/Work/Projects/"+self.model_name+"/vis/"

        self.schedule = [
            {"weights": None, 
             "num_batches": int(1e4),
             "weight_lr": float(0.01), 
             "decay_steps": 1,
             "decay_rate": 1.0,
             "staircase": False}
        ]

    
    def set_data_params(self, data_type):
        self.data_type = data_type
        if data_type.lower() == "vanhateren":
            self.model_name += "_vh"
            self.num_images = 150
            self.vectorize_data = True
            self.extract_patches = True

            # data preprocessing
            self.rescale_data = False
            self.whiten_data = True
            self.whiten_method = "ZCA"
            self.whiten_batch_size = 10
            self.num_neurons = 256

            # hyperparams
            self.num_groups = 64

            # logging
            self.cp_int = int(1e5)
            self.log_int = int(1e2)
            self.log_to_file = True
            self.gen_plot_int = int(2e4)

            self.schedule = [
                    {"weights": None,
                     "num_batches": int(1e3),
                     "weight_lr": float(0.001),
                     "decay_steps": int(5e5*0.8),
                     "decay_rate": 0.8,
                     "staircase": True}
                    ]
            


