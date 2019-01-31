import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      prior      [str] Prior for ICA - can be "laplacian" or "cauchy"
      batch_size [int] Number of images in a training batch
    """
    super(params, self).__init__()
    self.model_type = "ica"
    self.model_name = "ica"
    self.version = "0.0"
    self.num_images = 100
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "ZCA"
    self.lpf_data = True # only for ZCA/PCA
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0
    self.num_batches = int(5e5)
    self.batch_size = 50
    self.prior = "laplacian" #"cauchy"
    self.optimizer = "annealed_sgd"
    self.cp_int = 1000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_synth"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 1000
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "num_batches": int(5e5),
      "weight_lr": 0.001,
      "decay_steps": int(5e5*0.8),
      "decay_rate": 0.8,
      "staircase": True}]
  def set_data_params(self, data_type):
    if data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_patches = 50
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
