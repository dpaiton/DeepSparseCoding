import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      TODO
    """
    super(params, self).__init__()
    self.model_type = "ae"
    self.model_name = "ae_768"
    self.version = "1.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = True
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.batch_size = 100
    self.tie_decoder_weights = False
    self.norm_weights = False
    self.norm_w_init = False
    #Specify number of neurons for encoder
    #Last element in list is the size of the latent space
    #Decoder will automatically build the transpose of the encoder
    self.ae_layer_types = ["fc", "fc"]
    self.ae_output_channels = [512, 50]
    self.ae_patch_size = []
    self.ae_conv_strides = []
    self.ae_activation_functions = ["relu", "relu", "relu", "identity"]
    self.ae_dropout = [1.0]*4
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.schedule = [
      {"num_batches": int(3e5),
      "weights": None,
      "decay_mult": 0.01,
      "norm_mult": 0.00,
      "weight_lr": 1e-4,
      "decay_steps": int(3e5*0.4),
      "decay_rate": 0.9,
      "staircase": True,}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.rescale_data = False
      self.standardize_data = True
      self.ae_output_channels = [768, 256, 256, 128, 64]#[768]
      self.ae_layer_types = ["fc"]*len(self.ae_output_channels)
      self.optimizer = "annealed_sgd"
      self.batch_size = 100
      self.ae_activation_functions = ["relu"] * (2 * len(self.ae_layer_types) - 1) + ["identity"]
      self.ae_dropout = [0.5, 0.7, 0.7, 0.7, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0]#0.5, 1.0]#[0.35, 1.0]
      self.cp_int = int(5e5)
      self.gen_plot_int = int(5e5)
      self.norm_weights = False
      self.norm_w_init = False
      self.schedule = [
        {"num_batches": int(1e6),
        "weights": None,
        "decay_mult": 0.0,#1e-4,
        "norm_mult": 1e-4,#2e-4,#0.0,
        "weight_lr": 0.0001,#0.003,
        "decay_steps": int(6e5),
        "decay_rate": 0.5,
        "staircase": True,}]

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.vectorize_data = False
      self.standardize_data = True
      self.rescale_data = False
      self.ae_layer_types = ["conv"]
      self.ae_output_channels = [256]
      self.ae_patch_size = [(12, 12)]
      self.ae_conv_strides = [(1, 2, 2, 1)]
      self.optimizer = "adam"
      self.batch_size = 100
      self.ae_activation_functions = ["relu", "identity"]
      self.ae_dropout = [1.0] * 2
      self.cp_int = int(1e3)
      self.gen_plot_int = int(1e3)
      self.schedule = [
        {"num_batches": int(1e6),
        "weights": None,
        "decay_mult": 0.001,
        "norm_mult": 0.00,
        "weight_lr": 0.001,
        "decay_steps": int(800000),
        "decay_rate": 0.8,
        "staircase": True,}]

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.rescale_data = False
      self.vectorize_data = True
      self.whiten_data = True
      self.tf_standardize_data = False
      self.standardize_data = False
      self.whiten_data = True
      self.whiten_method = "FT"
      self.whiten_batch_size = 10
      self.lpf_data = False
      self.lpf_cutoff = 0.7
      self.extract_patches = True
      self.num_patches = 1e6
      self.patch_edge_size = 16
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      #self.ae_output_channels = [1536, 768, 256, 64]
      #self.ae_output_channels = [768, 256, 32]
      self.ae_output_channels = [768]
      self.ae_layer_types = ["fc"]*len(self.ae_output_channels)
      self.optimizer = "annealed_sgd"
      self.batch_size = 100
      self.ae_layer_types = ["fc"]*len(self.ae_output_channels)
      self.ae_activation_functions = ["relu"] * (2 * len(self.ae_layer_types) - 1) + ["identity"]
      self.ae_dropout = [0.3] * (len(self.ae_activation_functions) - 1) + [1.0]#[0.5, 0.5, 0.7, 1.0, 0.7, 0.7, 0.7, 1.0]
      self.log_int = 100
      self.cp_int = int(1e5)
      self.gen_plot_int = int(1e5)
      self.norm_weights = False
      self.norm_w_init = False
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(6e5)
        self.schedule[sched_idx]["weights"] = None
        self.schedule[sched_idx]["decay_mult"] = 2e-3
        self.schedule[sched_idx]["norm_mult"] = 1e-4
        self.schedule[sched_idx]["weight_lr"] = 1e-3
        self.schedule[sched_idx]["decay_steps"] = int(self.schedule[sched_idx]["num_batches"]*0.8)
        self.schedule[sched_idx]["decay_rate"] = 0.5
        self.schedule[sched_idx]["staircase"] = True

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    #TODO: allow for test_params to be list of params to be run individually
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 16
    self.tie_decoder_weights = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    # Test 1
    #self.ae_layer_types = ["fc", "fc", "fc"]
    #self.vectorize_data = True
    #self.ae_output_channels = [30, 20, 10]
    # Test 2
    #self.ae_layer_types = ["conv", "conv", "conv"]
    #self.vectorize_data = False
    #self.ae_output_channels = [30, 20, 10]
    #self.ae_patch_size = [(8,8), (4,4), (2,2)]
    #self.ae_conv_strides = [(1, 2, 2, 1), (1, 1, 1, 1), (1, 1, 1, 1)]
    # Test 3
    self.ae_layer_types = ["conv", "conv", "fc"]
    self.vectorize_data = False
    self.ae_output_channels = [30, 20, 10]
    self.ae_patch_size = [(8,8), (4,4)]
    self.ae_conv_strides = [(1, 2, 2, 1), (1, 1, 1, 1)]
    self.ae_activation_functions = ["relu"] * (2 * len(self.ae_output_channels) - 1) + ["identity"]
    self.ae_dropout = [1.0] * len(self.ae_activation_functions)
