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
    self.cp_int = int(1e5)
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "sigmoid_autoencoder"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_enc", "w_dec", "b_enc", "b_dec"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = int(1e5)
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "num_batches": int(1e5),
      "decay_mult": 0.008,
      "sparse_mult": 5.0, # How important is the sparse loss (tradeoff parameter)
      "target_act": 0.05, # Target firing rate for neurons
      "weight_lr": 0.002,
      "decay_steps": int(1e5*0.5),
      "decay_rate": 0.5,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = True
      self.center_data = False
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = False
      self.extract_patches = False
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_mult"] = 0.001
        self.schedule[sched_idx]["sparse_mult"] = 4.0
        self.schedule[sched_idx]["target_act"] = 0.5
        self.schedule[sched_idx]["weight_lr"] = 0.001
        self.schedule[sched_idx]["decay_steps"] = int(self.schedule[sched_idx]["decay_steps"]*0.8)
        self.schedule[sched_idx]["decay_rate"] = 0.90

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
        for sched_idx in range(len(self.schedule)):
          self.schedule[sched_idx]["num_batches"] = int(1e6)
          self.schedule[sched_idx]["decay_mult"] = 0.000
          self.schedule[sched_idx]["sparse_mult"] = 4.0
          self.schedule[sched_idx]["target_act"] = 0.5
          self.schedule[sched_idx]["weight_lr"] = 0.001
          self.schedule[sched_idx]["decay_steps"] = int(self.schedule[sched_idx]["decay_steps"]*0.4)

    elif data_type.lower() == "field":
        self.model_name += "_field"
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
        for sched_idx in range(len(self.schedule)):
          self.schedule[sched_idx]["decay_mult"] = 0.008
          self.schedule[sched_idx]["sparse_mult"] = 5.0
          self.schedule[sched_idx]["target_act"] = 0.05
          self.schedule[sched_idx]["weight_lr"] = 0.002

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.num_neurons = 768
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["decay_mult"] = 0.005
        self.schedule[sched_idx]["sparse_mult"] = 1.0
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["num_batches"] = int(1e5)

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    self.num_patches = 100
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
