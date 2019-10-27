import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "lca_subspace"
    self.model_name = "lca_subspace"
    self.version = "3.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = True
    self.standardize_data = False
    self.tf_standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False # only for ZCA
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.batch_size = 80
    self.num_neurons = 768
    self.num_groups = 192
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.norm_weights = True
    self.optimizer = "annealed_sgd"
    self.cp_int = int(1e5)
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["phi"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = int(1e5)
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "num_batches": int(1e6),
      "weight_lr": 0.01,
      "group_orth_mult": 0.01,
      "sparse_mult": 0.5,
      "decay_steps": int(1e6*0.8),
      "decay_rate": 0.7,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = True
      self.center_data = True
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = False
      self.lpf_data = False # only for ZCA
      self.lpf_cutoff = 0.7
      self.extract_patches = False
      self.batch_size = 100
      self.num_neurons = 1024
      self.num_groups = 256
      self.thresh_type = "soft"
      self.log_int = 100
      self.cp_int = int(5e5)
      self.gen_plot_int = int(1e5)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(7e5)
        self.schedule[sched_idx]["weight_lr"] = 0.08
        self.schedule[sched_idx]["group_orth_mult"] = 0.06
        self.schedule[sched_idx]["sparse_mult"] = 0.45
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.7

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 150
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = False
      self.center_data = False
      self.whiten_data = True
      self.whiten_method = "FT"
      self.whiten_batch_size = 10
      self.extract_patches = True
      self.num_patches = 1e6
      self.patch_edge_size = 16
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      self.num_neurons = 2560#1024
      self.num_groups = 640
      self.num_steps = 75
      self.cp_int = int(1e5)
      self.log_int = 100
      self.gen_plot_int = int(5e4)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(5e5)
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["group_orth_mult"] = 0.08
        self.schedule[sched_idx]["sparse_mult"] = 1.3
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.5

    elif data_type.lower() == "field":
      self.model_name += "_field"
      self.vectorize_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.extract_patches = True
      self.num_neurons = 768
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(2e5)
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["group_orth_mult"] = 0.002
        self.schedule[sched_idx]["sparse_mult"] = 0.1
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 8
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = True
      self.center_data = False
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = False
      self.lpf_data = False # only for ZCA
      self.lpf_cutoff = 0.7
      self.extract_patches = False
      self.num_neurons = 128
      self.num_groups = 32
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["group_orth_mult"] = 0.001
        self.schedule[sched_idx]["sparse_mult"] = 0.1
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    self.num_neurons = 100
    self.num_groups = 50
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
