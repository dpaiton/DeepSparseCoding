import os
from params.lca_params import params as lca_params

class params(lca_params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_pooling_units [int] number of 2nd layer units
    """
    super(params, self).__init__()
    self.model_type = "lca_pca"
    self.model_name = "lca_pca"
    self.version = "0.0"
    self.num_batches = int(1e5)
    self.batch_size = 100
    self.num_neurons = 768
    self.num_pooling_units = 192
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_step = int(1e6) # None gives latest checkpoint
    self.cp_load_name = "lca_pca_512_vh_ft_white"
    self.cp_load_ver = "0.0"
    #self.cp_load_var = ["phi"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True

    self.schedule = [
      {"weights": ["weights/w:0"],
      "sparse_mult": 1.0,
      "weight_lr": [0.01],
      "decay_steps": [int(self.num_batches*0.5)],
      "decay_rate": [0.8],
      "staircase": [True]}]
