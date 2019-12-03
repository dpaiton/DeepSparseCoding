import os
import numpy as np
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      memristor_type: the type of memristor for memristorize
      synthetic_noise: noise to create synthetic channels (e.g. upper/lower bounds for RRAM data with write verify)
    """
    super(params, self).__init__()
    self.model_type = "dae_mem"
    self.model_name = "dae_mem"
    self.version = "0.0"
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
    self.mirror_dec_architecture = True
    self.ae_layer_types = ["fc", "fc", "fc"]
    self.ae_enc_channels = [1500, 1000, 50]
    self.tie_dec_weights = False
    self.norm_weights = False
    self.w_init_type = "normal"
    self.ae_activation_functions = ["gdn", "gdn", "gdn", "gdn", "gdn", "identity"]
    self.ae_dropout = [1.0]*len(self.ae_activation_functions)
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
    self.memristor_data_loc = os.path.expanduser("~")+"/Work/DeepSparseCoding/memristor_data/Partial_Reset_PCM.pkl" 
    self.memristor_type = "rram"
    self.synthetic_noise = np.sqrt(2.0)
    self.mem_error_rate = 0.0
    self.optimizer = "sgd"
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
      "w_decay_mult": 0.03,
      "w_norm_mult": 0.0,
      "noise_variance_mult": 0.08,
      "weight_lr": 1e-5,
      "decay_steps": int(1e6*0.4),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.ae_layer_types = ["fc", "fc", "fc"]
      self.ae_enc_channels = [1568, 784, 50]
      self.ae_activation_functions = ["gdn", "gdn", "gdn", "gdn", "gdn", "identity"]
      self.ae_dropout = [1.0]*len(self.ae_activation_functions)

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
    self.vectorize_data = False
    self.tie_dec_weights = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.ae_dropout = [1.0]*4
    self.test_param_variants = [
      {"ae_layer_types":["conv", "conv"],
      "vectorize_data":False,
      "ae_enc_channels":[20, 10],
      "ae_conv_strides":[(1, 1, 1, 1), (1, 1, 1, 1)],
      "ae_patch_size":[(3, 3), (3, 3)],
      "ae_activation_functions":["gdn", "gdn", "gdn", "identity"]}]
