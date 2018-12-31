import os
import numpy as np
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      mle_step_size       [float] size of maximum likelihood estimator steps
      num_mle_steps       [int] number of maximum likelihood estimation steps for the entropy estimator
      num_triangles       [int] number of triangle kernels to use for the entropy estimator
      sigmoid_beta        [float] slope parameter for sigmoid activation function
      ramp_min            [float] minimum allowable value for latent variables where ramp loss = 0
      ramp_max            [float] maximum allowable value for latent variables where ramp loss = 0
      gdn_w_init_const    [float] initial value for the GDN weights
      gdn_b_init_const    [float] initial value for the GDN biases
      gdn_w_thresh_min    [float] minimum allowable value for GDN weights
      gdn_b_thresh_min    [float] minimum allowable value for GDN biases
      gdn_eps             [float] epsilon that will be added to the GDN denominator
    """
    super(params, self).__init__()
    self.model_type = "gdn_autoencoder"
    self.model_name = "gdn_autoencoder_ent"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = True
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.batch_size = 100
    self.num_neurons = 768
    self.mle_step_size = 0.01
    self.num_mle_steps = 30
    self.num_triangles = 30
    self.sigmoid_beta = 1.0
    self.ramp_min = -1.0
    self.ramp_max = 1.0
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
    self.cp_load_step = 1e4
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_enc", "b_enc", "w_dec", "b_dec", "w_gdn", "b_gdn", "w_igdn", "b_igdn"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.num_pixels = self.patch_edge_size**2

    self.schedule = [
      {"num_batches": int(1e6),
      "weights": ["w_enc", "b_enc", "w_dec", "b_dec", "w_gdn0", "b_gdn0", "w_igdn1", "b_igdn1"],
      "ent_mult": 0.1,
      "ramp_slope": 1.0,
      "decay_mult": 0.03,
      "noise_var_mult": 0.08,
      "triangle_centers": np.linspace(-1.0, 1.0, params().num_triangles),
      "weight_lr": [9e-5]*4+[7e-6]*4,
      "decay_steps": [int(1e6*0.4),]*8,
      "decay_rate": [0.8,]*8,
      "staircase": [True,]*8}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "vanhateren":
      self.model_name += "_vanhateren"
    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type=None):
    super(params, self).set_test_params(data_type)
    self.set_data_params(data_type)
