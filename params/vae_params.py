import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "vae"
    self.model_name = "test"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
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
    self.ae_enc_channels = [512, 50]
    self.ae_activation_functions = ["relu", "identity", "relu", "identity"]
    self.ae_dropout = [1.0]*4
    self.mirror_dec_architecture = True
    self.noise_level = 0.0 # std of noise added to the input data
    self.recon_loss_type = "mse" # or "cross-entropy"
    self.tie_dec_weights = False
    self.norm_weights = False
    self.w_init_type = "normal"
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
      "w_decay_mult": 0.0,
      "w_norm_mult": 0.0,
      "kld_mult": 1.0,
      "weight_lr": 0.0005,
      "decay_steps": int(3e5*0.8),
      "decay_rate": 0.8,
      "staircase": True,}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.optimizer = "adam"#"sgd"
      self.num_edge_pixels = 28
      self.num_data_channels = 1
      self.batch_size = 100
      self.log_int = 100
      self.cp_int = 5e5
      self.gen_plot_int = 1e4
      self.noise_level = 0.00
      self.center_data = False
      self.standardize_data = False
      self.tf_standardize_data = False
      self.vectorize_data = False
      self.mirror_dec_architecture = True
      self.ae_layer_types = ["conv", "conv", "fc"]
      self.ae_conv_strides = [(1, 2, 2, 1), (1, 1, 1, 1)]
      self.ae_patch_size = [(3, 3)]*2
      self.ae_enc_channels = [32, 64, 25]
      self.ae_activation_functions = ["lrelu", "lrelu", "sigmoid", "lrelu", "lrelu", "sigmoid"]
      self.ae_dropout = [1.0]*len(self.ae_activation_functions)
      self.recon_loss_type = "mse"
      for schedule_idx in range(len(self.schedule)):
        self.schedule[schedule_idx]["num_batches"] = int(1e5)#int(2e6)
        self.schedule[schedule_idx]["weight_lr"] = 1e-4
        self.schedule[schedule_idx]["kld_mult"] = 1.0
        self.schedule[schedule_idx]["w_decay_mult"] = 1e-3
        self.schedule[schedule_idx]["w_norm_mult"] = 0.0#2e-4
        self.schedule[schedule_idx]["decay_steps"] = int(1.0*self.schedule[schedule_idx]["num_batches"])
        self.schedule[schedule_idx]["decay_rate"] = 1.0

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_data_channels = 1

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["weights"] = None
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.ae_activation_functions = ["relu"] * 6
    self.ae_dropout = [1.0] * 6
    # Test 1
    self.test_param_variants = [
      {"vectorize_data":False,
      "tie_dec_weights":False,
      "ae_activation_functions":["relu"] * 4,
      "ae_dropout":[1.0] * 4,
      "mirror_dec_architecture":False,
      "ae_layer_types":["conv", "conv", "fc", "fc"],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1)],
      "ae_patch_size":[(3, 3)]*2,
      "ae_enc_channels":[32, 64, 25],
      "ae_dec_channels":[int(self.num_edge_pixels**2)]}]
    # Test 2
    self.test_param_variants += [
      {"vectorize_data":False,
      "tie_dec_weights":False,
      "mirror_dec_architecture":True,
      "ae_layer_types":["conv", "conv", "fc"],
      "ae_enc_channels":[30, 20, 10],
      "ae_patch_size":[(8,8), (4,4)],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1)]}]
