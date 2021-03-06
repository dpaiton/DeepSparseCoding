from DeepSparseCoding.tf1x.params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
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
    self.tie_dec_weights = False
    self.norm_weights = False
    self.w_init_type = "normal"
    self.mirror_dec_architecture = True
    self.ae_layer_types = ["fc", "fc"]
    self.ae_enc_channels = [512, 50]
    self.ae_patch_size = []
    self.ae_conv_strides = []
    self.ae_activation_functions = ["relu", "relu", "relu", "identity"]
    self.ae_dropout = [1.0]*4
    self.optimizer = "sgd"
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
      "w_decay_mult": 0.01,
      "w_norm_mult": 0.00,
      "weight_lr": 1e-4,
      "decay_steps": int(3e5*0.4),
      "decay_rate": 0.9,
      "staircase": True,}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.num_edge_pixels = 28
      self.num_data_channels = 1
      self.rescale_data = False
      self.standardize_data = True
      self.tf_standardize_data = False
      self.center_data = False
      self.optimizer = "sgd"
      self.batch_size = 100
      self.vectorize_data = True
      self.tie_dec_weights = False
      self.mirror_dec_architecture = False
      self.ae_enc_channels = [768, 256, 256, 128, 64]
      self.ae_dec_channels = self.ae_enc_channels[::-1][1:]+[784]
      self.ae_layer_types = ["fc"]*2*len(self.ae_enc_channels)
      self.ae_activation_functions = ["lrelu"] * (len(self.ae_layer_types) - 1) + ["identity"]
      self.ae_dropout = [0.5, 0.7, 0.7, 0.7, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0]
      self.log_int = 100
      self.cp_int = int(5e5)
      self.gen_plot_int = int(5e5)
      self.norm_weights = False
      self.w_init_type = "normal"
      self.schedule = [
        {"num_batches": int(1e6),
        "weights": None,
        "w_decay_mult": 0.0,#1e-4,
        "w_norm_mult": 1e-4,#2e-4,#0.0,
        "weight_lr": 0.002,
        "decay_steps": int(6e5),
        "decay_rate": 0.5,
        "staircase": True,}]

    elif data_type.lower() == "field":
      self.model_name += "_field"
      self.num_edge_pixels = 16
      self.num_data_channels = 1
      self.extract_patches = True
      self.patch_edge_size = 16
      self.num_patches = 1e4
      self.overlapping_patches = True
      self.randomize_patches = True
      self.vectorize_data = True
      self.standardize_data = True
      self.rescale_data = False
      self.ae_layer_types = ["fc", "fc"]
      self.ae_enc_channels = [256, 16]
      self.tie_dec_weights = True
      self.mirror_dec_architecture = True
      self.optimizer = "adam"
      self.batch_size = 100
      self.ae_activation_functions = ["lrelu", "sigmoid"]
      self.ae_dropout = [1.0] * len(self.ae_activation_functions)
      self.cp_int = int(1e3)
      self.gen_plot_int = int(1e3)
      self.schedule = [
        {"num_batches": int(1e6),
        "weights": None,
        "w_decay_mult": 0.001,
        "w_norm_mult": 0.00,
        "weight_lr": 0.001,
        "decay_steps": int(800000),
        "decay_rate": 0.8,
        "staircase": True,}]

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.num_edge_pixels = 32
      self.num_data_channels = 3
      self.vectorize_data = False
      self.standardize_data = True
      self.rescale_data = False
      self.ae_layer_types = ["conv"]
      self.ae_enc_channels = [256]
      self.ae_patch_size = [(8, 8)]
      self.ae_conv_strides = [(1, 1, 1, 1)]
      self.tie_dec_weights = False
      self.mirror_dec_architecture = True
      self.optimizer = "adam"
      self.batch_size = 100
      self.ae_activation_functions = ["relu", "identity"]
      self.ae_dropout = [1.0] * 2
      self.cp_int = int(1e3)
      self.gen_plot_int = int(1e3)
      self.schedule = [
        {"num_batches": int(1e6),
        "weights": None,
        "w_decay_mult": 0.001,
        "w_norm_mult": 0.00,
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
      self.num_edge_pixels = self.patch_edge_size
      self.num_data_channels = 1
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      self.batch_size = 100
      self.tie_dec_weights = False
      self.mirror_dec_architecture = True
      self.ae_enc_channels = [768, 256, 64]
      self.ae_layer_types = ["fc"]*(2*len(self.ae_enc_channels))
      self.optimizer = "sgd"
      self.ae_activation_functions = ["relu"] * (2 * len(self.ae_layer_types) - 1) + ["identity"]
      self.ae_dropout = [0.3] * (len(self.ae_activation_functions) - 1) + [1.0]#[0.5, 0.5, 0.7, 1.0, 0.7, 0.7, 0.7, 1.0]
      self.log_int = 100
      self.cp_int = int(1e5)
      self.gen_plot_int = int(1e5)
      self.norm_weights = False
      self.w_init_type = "normal"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(6e5)
        self.schedule[sched_idx]["weights"] = None
        self.schedule[sched_idx]["w_decay_mult"] = 2e-3
        self.schedule[sched_idx]["w_norm_mult"] = 1e-4
        self.schedule[sched_idx]["weight_lr"] = 1e-3
        self.schedule[sched_idx]["decay_steps"] = int(self.schedule[sched_idx]["num_batches"]*0.8)
        self.schedule[sched_idx]["decay_rate"] = 0.5
        self.schedule[sched_idx]["staircase"] = True

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
    # Test 1
    self.test_param_variants = [
      {"vectorize_data":True,
      "tie_dec_weights":True,
      "mirror_dec_architecture":False,
      "ae_activation_functions":["relu"]*6,
      "ae_dropout":[1.0]*6,
      "ae_layer_types":["fc", "fc", "fc", "fc", "fc", "fc"],
      "ae_enc_channels":[30, 20, 10],
      "ae_dec_channels":[20, 30, self.num_edge_pixels**2*self.num_data_channels]}]
    # Test 2
    self.test_param_variants += [
      {"vectorize_data":False,
      "tie_dec_weights":True,
      "mirror_dec_architecture":False,
      "ae_activation_functions":["relu"]*6,
      "ae_dropout":[1.0]*6,
      "ae_layer_types":["conv", "conv", "conv", "conv", "conv", "conv"],
      "ae_enc_channels":[30, 20, 10],
      "ae_dec_channels":[20, 30, 1],
      "ae_patch_size":[(8,8), (4,4), (2,2), (2,2), (4,4), (8,8)],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1), (1, 1, 1, 1),
      (1, 1, 1, 1), (1, 1, 1, 1), (1, 2, 2, 1)]}]
    # Test 3
    self.test_param_variants += [
      {"vectorize_data":False,
      "tie_dec_weights":False,
      "mirror_dec_architecture":True,
      "ae_activation_functions":["relu"]*6,
      "ae_dropout":[1.0]*6,
      "ae_layer_types":["conv", "conv", "fc"],
      "ae_enc_channels":[30, 20, 10],
      "ae_patch_size":[(8,8), (4,4)],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1)]}]
    # Test 4
    self.test_param_variants += [
      {"vectorize_data":False,
      "tie_dec_weights":False,
      "mirror_dec_architecture":False,
      "ae_layer_types":["conv", "conv", "conv", "fc"],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1), (1, 1, 1, 1)],
      "ae_patch_size":[(3, 3)]*3,
      "ae_enc_channels":[32, 64, 25],
      "ae_activation_functions":["relu"]*4,
      "ae_dropout":[1.0]*4,
      "ae_dec_channels":[self.num_edge_pixels**2*self.num_data_channels]}]
