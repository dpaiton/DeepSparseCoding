import os
import numpy as np
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      mle_step_size       [float] size of maximum likelihood estimator steps
      num_mle_steps       [int] number of maximum likelihood estimation steps for the entropy estimator
      num_triangles       [int] number of triangle kernels to use for the entropy estimator
      bounds_slope        [float] slope for out of bounds loss (two relus back to back)
      latent_min          [float] minimum allowable value for latent variables where ramp loss = 0
      latent_max          [float] maximum allowable value for latent variables where ramp loss = 0
      mle_step_size       [float] size of maximimum likelihood estimator steps
      num_mle_steps       [int] number of max likelihood estimation steps for the entropy estimator
      num_quant_bins      [int] number of bins you want for quantization
      gdn_w_init_const    [float] initial value for the GDN weights
      gdn_b_init_const    [float] initial value for the GDN biases
      gdn_w_thresh_min    [float] minimum allowable value for GDN weights
      gdn_b_thresh_min    [float] minimum allowable value for GDN biases
      gdn_eps             [float] epsilon that will be added to the GDN denominator
    """
    super(params, self).__init__()
    self.model_type = "dae"
    self.model_name = "dae_test"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = False
    self.rescale_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.whiten_method = "FT"
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.batch_size = 100
    self.output_channels = [1500, 1000, 50]
    self.tie_decoder_weights = False
    self.conv = False
    self.activation_functions = ["gdn", "gdn", "gdn", "gdn", "gdn", "identity"]
    self.dropout = [1.0]*6
    self.num_triangles = 30
    self.mle_step_size = 0.01
    self.num_mle_steps = 30
    self.num_quant_bins = 256 # number of pixel bins
    self.bounds_slope = 1.0
    self.latent_min = -1.0
    self.latent_max = 1.0
    self.gdn_w_init_const = 0.1
    self.gdn_b_init_const = 0.1
    self.gdn_w_thresh_min = 1e-6
    self.gdn_b_thresh_min = 1e-6
    self.gdn_eps = 1e-6
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "gdn_autoencoder"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.num_pixels = self.patch_edge_size**2
    self.schedule = [
      {"num_batches": int(1e6),
      "weights": None,
      "entropy_mult": 0.1,
      "decay_mult": 0.03,
      "noise_variance_mult": 0.08,
      "weight_lr": 1e-5,
      "decay_steps": int(1e6*0.4),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.conv = False
      self.output_channels = [768, 512, 50]
      self.activation_functions = ["gdn", "gdn", "sigmoid", "gdn", "gdn", "identity"]
      self.dropout = [1.0]*len(self.activation_functions)
      self.cp_int = int(1e5)
      self.gen_plot_int = int(1e5)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["entropy_mult"] = 0.01
        self.schedule[sched_idx]["decay_mult"] = 0.01
        self.schedule[sched_idx]["noise_variance_mult"] = 0.01
        self.schedule[sched_idx]["weight_lr"] = 1e-3

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vanhateren"

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 11
    self.num_edge_pixels = 8
    self.tie_decoder_weights = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.output_channels = [20, 10]
    self.conv = True
    self.conv_strides = [(1, 1, 1, 1), (1, 1, 1, 1)]
    self.patch_size_y = [3.0, 3.0]
    self.patch_size_x = self.patch_size_y
    self.activation_functions = ["gdn", "gdn", "gdn", "identity"]
    self.dropout = [1.0]*4
    self.vectorize_data = False
