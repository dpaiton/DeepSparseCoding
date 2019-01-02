import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_groups [int] number of 2nd layer units
    """
    super(params, self).__init__()
    self.model_type = "sigmoid_autoencoder"
    self.model_name = "sigmoid_autoencoder"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
    self.center_data = False
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
    self.batch_size = 200
    self.num_neurons = 768
    self.optimizer = "annealed_sgd"#"adam"
    self.cp_int = 100000
    self.max_cp_to_keep = 1
    self.cp_load = True
    self.cp_load_name = "sigmoid_autoencoder"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_enc", "w_dec", "b_enc", "b_dec"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 100000
    self.save_plots = True
    self.num_batches = int(1e6)
    self.schedule = [
      {"weights": ["w_enc", "b_enc", "w_dec", "b_dec"],
      "decay_mult": 0.008,
      "sparse_mult": 5.0, # How important is the sparse loss (tradeoff parameter)
      "target_act": 0.05, # Target firing rate for neurons
      "weight_lr": [0.002,]*4,
      "decay_steps": [int(self.num_batches*0.5),]*4,
      "decay_rate": [0.5,]*4,
      "staircase": [True,]*4}]

  def set_data_params(data_type):
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = True
      self.center_data = False
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = False
      self.whiten_method = "FT"
      self.lpf_data = False
      self.lpf_cutoff = 0.7
      self.extract_patches = False
      self.schedule["decay_mult"] = 0.005
      self.schedule["sparse_mult"] = 0.01
      self.schedule["target_act"] = 0.09
      self.schedule["weight_lr"] = [0.01,]*4

    elif data_type.lower() == "vanhateren":
        self.model_name += "_vh"
        self.num_images = 150
        self.vectorize_data = True
        self.norm_data = False
        self.rescale_data = False
        self.center_data = False
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
        self.num_neurons = 768
        self.schedule["decay_mult"] = 0.008
        self.schedule["sparse_mult"] = 5.0
        self.schedule["target_act"] = 0.05
        self.schedule["weight_lr"] = [0.002,]*4
