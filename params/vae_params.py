import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      TODO
    """
    super(params, self).__init__()
    self.model_type = "vae"
    self.model_name = "vae_deep_denoising"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = True
    self.center_data = False
    self.standardize_data = False
    self.tf_standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.batch_size = 100
    # Specify number of neurons for encoder
    # Last element in list is the size of the latent space
    # Decoder will automatically build the transpose of the encoder
    self.ae_layer_types = ["fc", "fc"]
    self.ae_output_channels = [512, 50]
    self.ae_activation_functions = ["relu", "identity", "relu", "identity"]
    self.ae_dropout = [1.0]*4
    self.noise_level = 0.0 # std of noise added to the input data
    self.recon_loss_type = "mse" # or "cross-entropy"
    self.tie_decoder_weights = False
    self.norm_weights = False
    self.norm_w_init = False
    self.optimizer = "adam"
    self.cp_int = 1e4
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 1e4
    self.save_plots = True
    self.schedule = [
      {"num_batches": int(3e5),
      "weights": None,
      "decay_mult": 0.0,
      "norm_mult": 0.0,
      "kld_mult": 1.0,
      "weight_lr": 0.0005,
      "decay_steps": int(3e5*0.8),
      "decay_rate": 0.8,
      "staircase": True,}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.optimizer = "adam"#"annealed_sgd"#"adam"
      self.batch_size = 100
      self.log_int = 100
      self.cp_int = 5e5
      self.gen_plot_int = 5e5
      self.noise_level = 0.08
      self.ae_layer_types = ["fc", "fc", "fc"]
      self.ae_output_channels = [768, 256, 64]#[768]
      self.recon_loss_type = "mse" # "mse" or "crossentropy"
      self.ae_activation_functions = ["lrelu", "lrelu", "identity", "lrelu", "lrelu", "relu"]#["relu", "relu"]
      self.ae_dropout = [0.5, 0.8, 1.0, 0.8, 0.8, 1.0]#[1.0, 1.0]
      for schedule_idx in range(len(self.schedule)):
        self.schedule[schedule_idx]["num_batches"] = int(2e6)
        self.schedule[schedule_idx]["weight_lr"] = 8e-4
        self.schedule[schedule_idx]["kld_mult"] = 1.0
        self.schedule[schedule_idx]["decay_mult"] = 0.0
        self.schedule[schedule_idx]["norm_mult"] = 2e-4
        self.schedule[schedule_idx]["decay_steps"] = int(0.7*self.schedule[schedule_idx]["num_batches"])
        self.schedule[schedule_idx]["decay_rate"] = 0.9

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
    self.vectorize_data = True
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.ae_activation_functions = ["relu", "identity", "relu", "identity"]
    self.ae_dropout = [1.0]*len(self.ae_activation_functions)
    self.ae_output_channels = [20, 10]
    # Test 1
    self.ae_layer_types = ["fc", "fc"]
    # Test 2
    #self.ae_layer_types = ["conv", "conv"]
    #self.ae_conv_strides = [(1, 1, 1, 1), (1, 1, 1, 1)]
    #self.ae_patch_size = [(3, 3), (3, 3)]
