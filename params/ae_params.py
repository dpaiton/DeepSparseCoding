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
    self.model_name = "ae"
    self.version = "0.0"
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
    #Specify number of neurons for encoder
    #Last element in list is the size of the latent space
    #Decoder will automatically build the transpose of the encoder
    self.output_channels = [512, 50]
    self.tie_decoder_weights = False
    self.activation_functions = ["relu", "relu", "relu", "identity"]
    self.dropout = [1.0]*4
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
      "weight_lr": 1e-4,
      "decay_steps": int(3e5*0.4),
      "decay_rate": 0.9,
      "staircase": True,}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.output_channels = [768, 50]
      self.optimizer = "annealed_sgd"
      self.batch_size = 100
      self.activation_functions = ["lrelu", "lrelu", "lrelu", "sigmoid"]
      self.dropout = [0.5, 0.5, 0.5, 1.0]
      self.cp_int = int(1e3)
      self.gen_plot_int = int(1e3)
      self.schedule = [
        {"num_batches": int(5e3),
        "weights": None,
        "decay_mult": 0.0007,
        "weight_lr": 0.005,
        "decay_steps": int(3e3),
        "decay_rate": 0.5,
        "staircase": True,}]

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
    self.batch_size = 10
    self.num_edge_pixels = 8
    self.tie_decoder_weights = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.output_channels = [20, 10]
    self.activation_functions = ["relu", "relu", "relu", "identity"]
    self.dropout = [1.0]*4
