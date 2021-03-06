from DeepSparseCoding.tf1x.params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "vae"
    self.model_name = "scvae"
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
    self.noise_level = 0.0 # std of noise added to the input data
    self.prior_params = {
      "posterior_prior":"gauss_gasus",
      "gauss_mean":0.0,
      "gauss_std":1.0,
      "cauchy_location":0.0,
      "cauchy_scale":0.2,
      "laplace_scale":1.0
    }
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
      self.ae_layer_types = ["conv", "conv"]
      self.ae_conv_strides = [(1, 2, 2, 1), (1, 1, 1, 1)]
      self.ae_patch_size = [(3, 3)]*2
      self.ae_enc_channels = [32, 64]
      self.ae_activation_functions = ["lrelu", "lrelu", "lrelu", "lrelu", "sigmoid"]
      self.ae_dropout = [1.0]*len(self.ae_activation_functions)
      self.vae_mean_layer_types = ["fc"]
      self.vae_mean_channels = [25]
      self.vae_mean_activation_functions = ["sigmoid"]*len(self.vae_mean_layer_types)
      self.vae_mean_dropout = [1.0]*len(self.vae_mean_layer_types)
      self.vae_mean_conv_strides = []
      self.vae_mean_patch_size = []
      self.vae_var_layer_types = ["fc"]
      self.vae_var_channels = [25]
      self.vae_var_activation_functions = ["sigmoid"]*len(self.vae_var_layer_types)
      self.vae_var_dropout = [1.0]*len(self.vae_var_layer_types)
      self.vae_var_conv_strides = []
      self.vae_var_patch_size = []
      self.prior_params = {
        "posterior_prior":"gauss_gasus",
        "gauss_mean":0.0,
        "gauss_std":1.0,
        "cauchy_location":0.0,
        "cauchy_scale":0.2,
        "laplace_scale":1.0
      }
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)#int(2e6)
        self.schedule[sched_idx]["weight_lr"] = 1e-4
        self.schedule[sched_idx]["kld_mult"] = 1.0
        self.schedule[sched_idx]["w_decay_mult"] = 1e-3
        self.schedule[sched_idx]["w_norm_mult"] = 0.0#2e-4
        self.schedule[sched_idx]["decay_steps"] = int(1.0*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 1.0

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
      self.batch_size = 32
      self.tie_dec_weights = False
      self.mirror_dec_architecture = False

      self.ae_layer_types = ["fc", "fc"]
      self.ae_enc_channels = [128]
      self.ae_dec_channels = [self.num_edge_pixels**2*self.num_data_channels]
      self.ae_activation_functions = ["lrelu", "identity"]
      self.ae_dropout = [1.0]*len(self.ae_layer_types)
      self.ae_conv_strides = []
      self.ae_patch_size = []

      self.vae_mean_layer_types = ["fc", "fc", "fc"]
      self.vae_mean_channels = [256, 512, 169]
      self.vae_mean_activation_functions = ["lrelu", "lrelu", "lrelu"]
      self.vae_mean_dropout = [1.0]*len(self.vae_mean_layer_types)
      self.vae_mean_conv_strides = []
      self.vae_mean_patch_size = []

      self.vae_var_layer_types = ["fc", "fc", "fc"]
      self.vae_var_channels = [256, 512, 169]
      self.vae_var_activation_functions = ["lrelu", "lrelu", "sigmoid"]
      self.vae_var_dropout = [1.0]*len(self.vae_var_layer_types)
      self.vae_var_conv_strides = []
      self.vae_var_patch_size = []

      self.optimizer = "adam"
      self.log_int = 100
      self.cp_int = int(5e5)
      self.gen_plot_int = int(1e5)
      self.norm_weights = False
      self.w_init_type = "normal"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["weights"] = None
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["weight_lr"] = 1e-3
        self.schedule[sched_idx]["kld_mult"] = 1.0
        self.schedule[sched_idx]["w_decay_mult"] = 0.0
        self.schedule[sched_idx]["w_norm_mult"] = 0.0
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
    self.vae_mean_layer_types = ["fc"]
    self.vae_mean_channels = [50]
    self.vae_mean_activation_functions = ["relu"]*len(self.vae_mean_layer_types)
    self.vae_mean_dropout = [1.0]*len(self.vae_mean_layer_types)
    self.vae_mean_conv_strides = []
    self.vae_mean_patch_size = []
    self.vae_var_layer_types = ["fc"]
    self.vae_var_channels = [50]
    self.vae_var_activation_functions = ["relu"]*len(self.vae_var_layer_types)
    self.vae_var_dropout = [1.0]*len(self.vae_var_layer_types)
    self.vae_var_conv_strides = []
    self.vae_var_patch_size = []
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
      "posterior_prior":"gauss_gauss",#laplacian",
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
      "posterior_prior":"gauss_gauss",
      "ae_layer_types":["conv", "conv", "fc"],
      "ae_enc_channels":[30, 20, 10],
      "ae_patch_size":[(8,8), (4,4)],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1)]}]
    # Test 3
    self.test_param_variants += [
      {"vectorize_data":False,
      "tie_dec_weights":False,
      "mirror_dec_architecture":True,
      "posterior_prior":"cauchy_cauchy",
      "ae_layer_types":["conv", "conv", "fc"],
      "ae_enc_channels":[30, 20, 10],
      "ae_patch_size":[(8,8), (4,4)],
      "ae_conv_strides":[(1, 2, 2, 1), (1, 1, 1, 1)]}]
