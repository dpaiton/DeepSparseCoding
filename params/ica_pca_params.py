import os
import numpy as np
from params.ica_params import params as ica_params

class params(ica_params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_pooling_units [int] number of 2nd layer units
    """
    super(params, self).__init__()
    self.model_type = "ica_pca"
    self.model_name = "ica_pca"
    self.version = "0.0"
    self.num_images = 50
    self.vectorize_data = True
    self.norm_data = False
    self.whiten_data = True
    self.contrast_normalize = False
    self.extract_patches = True
    self.num_patches = 1e5
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 1e-6
    self.num_batches = int(1e5)
    self.batch_size = 100
    self.prior = "laplacian" # "cauchy"
    self.optimizer = "annealed_sgd"
    self.num_pooling_units = 50
    self.cp_int = 10000
    self.max_cp_to_keep = 2
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = 150000
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["a"]
    self.log_int = 500
    self.log_to_file = True
    self.gen_plot_int = 500
    self.save_plots = True
    self.schedule = [
      {"weights": ["a"],
      "weight_lr": [0.01],
      "decay_steps": [int(self.num_batches*0.8)],
      "decay_rate": [0.7],
      "staircase": [True]}]
