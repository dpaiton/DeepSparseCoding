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
      {"weights": ["weights/w_analy:0"],
      "num_batches": int(1e3), #int(5e5),
      "weight_lr": 0.001,
      "decay_steps": int(5e5*0.8),
      "decay_rate": 0.8,
      "staircase": True}]


  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.version = "1"
      self.num_images = 150
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = False
      self.center_data = True
      self.whiten_data = True
      self.whiten_method = "ZCA"
      self.extract_patches = True
      self.whiten_batch_size = 10
      self.num_patches = int(5*1e4)
      self.batch_size = 5000
      self.patch_edge_size = 16
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      self.orthonorm_weights = False
      
      self.group_size = 4
      self.num_neurons = 256
      self.num_pixels = 256
      self.num_groups = self.num_neurons // self.group_size

      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.5
