import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      output_channels [list of ints]
      activation_functions [list of strs]
    """
    super(params, self).__init__()
    self.model_type = "sae"
    self.model_name = "sae_768"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
    self.center_data = False
    self.standardize_data = True
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
    self.tie_decoder_weights = False
    self.norm_weights = False
    self.norm_w_init = False
    self.batch_size = 100
    self.layer_types = ["fc"]
    self.output_channels = [768]
    self.patch_size = []
    self.conv_strides = []
    self.activation_functions = ["sigmoid", "identity"]
    self.dropout = [1.0]*2
    self.optimizer = "annealed_sgd"#"adam"
    self.cp_int = int(1e5)
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "sae"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = None
    self.log_int = 500
    self.log_to_file = True
    self.gen_plot_int = int(1e5)
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "num_batches": int(1e5),
      "decay_mult": 0.01,
      "norm_mult": 0.0,
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
      self.cp_int = int(5e5)
      self.gen_plot_int = int(5e5)
      self.batch_size = 100
      self.vectorize_data = True
      self.rescale_data = True
      self.standardize_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.output_channels = [768]
      self.norm_w_init = False
      self.activation_functions = ["sigmoid", "relu"]
      self.dropout = [1.0]*2*len(self.output_channels)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["weight_lr"] = 0.0002
        self.schedule[sched_idx]["decay_mult"] = 0.00
        self.schedule[sched_idx]["norm_mult"] = 2e-3
        self.schedule[sched_idx]["target_act"] = 0.20
        self.schedule[sched_idx]["sparse_mult"] = 0.10
        self.schedule[sched_idx]["decay_steps"] = int(0.6*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.50

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.vectorize_data = False
      self.standardize_data = True
      self.rescale_data = False
      self.cp_int = int(1e4)
      self.gen_plot_int = int(1e4)
      self.batch_size = 100
      self.whiten_data = False
      self.extract_patches = False
      self.layer_types = ["conv"]
      self.output_channels = [256]
      self.patch_size = [(12, 12)]
      self.conv_strides = [(1, 2, 2, 1)]
      self.activation_functions = ["sigmoid", "identity"]
      self.optimizer = "adam"
      self.dropout = [1.0]*2*len(self.output_channels)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["weight_lr"] = 0.001
        self.schedule[sched_idx]["decay_mult"] = 0.05
        self.schedule[sched_idx]["norm_mult"] = 0.0
        self.schedule[sched_idx]["target_act"] = 0.10
        self.schedule[sched_idx]["sparse_mult"] = 100.0
        self.schedule[sched_idx]["decay_steps"] = int(0.6*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.50

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 150
      self.batch_size = 100
      self.vectorize_data = True
      self.rescale_data = False
      self.standardize_data = True
      self.whiten_data = True
      self.whiten_method = "FT"
      self.whiten_batch_size = 10
      self.extract_patches = True
      self.output_channels = [768]
      self.activation_functions = ["sigmoid", "identity"]
      self.dropout = [1.0]*2
      self.log_int = 100
      self.cp_int = int(5e5)
      self.gen_plot_int = int(2e5)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["decay_mult"] = 0.00
        self.schedule[sched_idx]["norm_mult"] = 0.01
        self.schedule[sched_idx]["target_act"] = 0.20
        self.schedule[sched_idx]["sparse_mult"] = 0.1
        self.schedule[sched_idx]["weight_lr"] = 0.003
        self.schedule[sched_idx]["decay_steps"] = int(self.schedule[sched_idx]["num_batches"]*0.5)
        self.schedule[sched_idx]["decay_rate"] = 0.50

    elif data_type.lower() == "field":
      self.model_name += "_field"
      self.vectorize_data = True
      self.rescale_data = False
      self.standardize_data = True
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
      self.output_channels = [768]
      self.activation_functions = ["sigmoid", "identity"]
      self.dropout = [1.0]*2
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["decay_mult"] = 0.008
        self.schedule[sched_idx]["norm_mult"] = 0.0
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
      self.output_channels = [768]
      self.activation_functions = ["sigmoid", "identity"]
      self.dropout = [1.0]*2
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_mult"] = 0.03
        self.schedule[sched_idx]["sparse_mult"] = 0.15
        self.schedule[sched_idx]["target_act"] = 0.2
        self.schedule[sched_idx]["weight_lr"] = 1e-3

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
      self.schedule[sched_idx]["decay_mult"] = 0.03
      self.schedule[sched_idx]["sparse_mult"] = 0.15
      self.schedule[sched_idx]["target_act"] = 0.2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.layer_types = ["fc"]
    self.vectorize_data = True
    self.output_channels = [30]
    self.activation_functions = ["sigmoid", "identity"]
    self.dropout = [1.0]*len(self.activation_functions)
