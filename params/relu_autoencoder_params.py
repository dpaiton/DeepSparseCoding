import os
import numpy as np
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      mle_step_size [float] size of maximum likelihood estimator steps
      num_mle_steps [int] number of maximum likelihood estimation steps for the entropy estimator
      num_triangles [int] number of triangle kernels to use for the entropy estimator
      sigmoid_beta  [float] slope parameter for sigmoid activation function
    """
    super(params, self).__init__()
    self.model_type = "relu_autoencoder"
    self.model_name = "relu_autoencoder"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = True
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.num_batches = int(1e5)
    self.batch_size = 150
    self.num_neurons = 768
    self.mle_step_size = 0.01
    self.num_mle_steps = 15
    self.num_triangles = 20
    self.sigmoid_beta = 1.0
    self.optimizer = "adam" #"annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "relu_autoencoder"
    self.cp_load_step = 1e4
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_enc", "b_enc", "w_dec", "b_dec"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True

    self.schedule = [
      {"weights": ["w_enc", "b_enc", "w_dec", "b_dec"],
      "ent_mult": 10.00,
      "decay_mult": 0.001,
      "noise_var_mult": 0.0,
      "triangle_centers": np.linspace(-1.0, 1.0, self.num_triangles),
      "weight_lr": [1e-4,]*4,
      "decay_steps": [int(self.num_batches*0.8),]*4,
      "decay_rate": [0.8,]*4,
      "staircase": [True,]*4}]
