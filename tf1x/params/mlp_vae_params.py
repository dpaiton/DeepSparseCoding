from DeepSparseCoding.tf1x.params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of LCA neurons
    """
    super(params, self).__init__()
    self.model_type = "mlp_vae"
    self.model_name = "mlp_vae"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False # FT whitening already does LPF
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.batch_size = 100
    # VAE Params
    self.ae_layer_types = ["fc"]
    self.mirror_dec_architecture = True
    self.ae_enc_channels = [768]
    self.ae_patch_size = []
    self.ae_conv_strides = []
    self.ae_activation_functions = ["lrelu", "sigmoid"]
    self.ae_dropout = [1.0]*2*len(self.ae_enc_channels)
    self.vae_mean_layer_types = ["fc"]
    self.vae_mean_channels = [768]
    self.vae_mean_activation_functions = ["lrelu"]*len(self.vae_mean_layer_types)
    self.vae_mean_dropout = [1.0]*len(self.vae_mean_layer_types)
    self.vae_mean_conv_strides = []
    self.vae_mean_patch_size = []
    self.vae_var_layer_types = ["fc"]
    self.vae_var_channels = [768]
    self.vae_var_activation_functions = ["sigmoid"]*len(self.vae_var_layer_types)
    self.vae_var_dropout = [1.0]*len(self.vae_var_layer_types)
    self.vae_var_conv_strides = []
    self.vae_var_patch_size = []
    self.prior_params = {
      "posterior_prior":"gauss_gasus",
      "gauss_prior_mean":0.0,
      "gauss_prior_std":1.0
    }
    self.noise_level = 0.01 # std of noise added to the input data
    self.tie_dec_weights = False
    self.norm_weights = False
    self.w_init_type = "normal"
    self.optimizer = "adam"
    # MLP Params
    self.train_on_recon = True # if False, train on LCA latent activations
    self.num_val = 10000
    self.num_labeled = 50000
    self.num_classes = 10
    self.mlp_layer_types = ["fc", "fc", "fc"]
    self.mlp_output_channels = [300, 500, self.num_classes]
    self.mlp_activation_functions = ["lrelu", "lrelu", "sigmoid"]
    self.mlp_patch_size = []
    self.mlp_conv_strides = []
    self.batch_norm = [None, None, None]
    self.mlp_dropout = [1.0, 1.0, 1.0]
    self.max_pool = [False, False, False]
    self.max_pool_ksize = [None, None, None]
    self.max_pool_strides = [None, None, None]
    self.lrn = [None]*len(self.mlp_output_channels)
    self.mlp_decay_mult = 0
    self.mlp_norm_mult = 1e-4
    #Adversarial params
    self.adversarial_num_steps = 40
    self.adversarial_attack_method = "kurakin_untargeted"
    self.adversarial_step_size = 0.01
    self.adversarial_max_change = 0.3
    # DEPRECATE self.adversarial_target_method = "random" #Not used if attack_method is untargeted
    self.adversarial_clip = True
    #TODO get these params from other params
    self.adversarial_clip_range = [0.0, 1.0]
    #Tradeoff in carlini attack between input pert and target
    self.carlini_recon_mult = 1
    # Others
    self.cp_int = 10000
    self.val_on_cp = True
    self.eval_batch_size = 100
    self.max_cp_to_keep = 1
    self.cp_load = True
    self.cp_load_name = "vae_mnist"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = [
        "vae/w_enc_1_std:0",
        "vae/b_enc_1_std:0",
        "vae/layer0/w_0:0",
        "vae/layer0/b_0:0",
        "vae/layer1/w_1:0",
        "vae/layer1/b_1:0",
        ]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.schedule = [
      #Training VAE
      #{"weights": None,
      #"train_vae": True,
      #"num_batches": int(3e5),
      #"w_decay_mult": 0.0,
      #"w_norm_mult": 0.0,
      #"kld_mult": 1/self.batch_size,
      #"weight_lr": 0.001,
      #"decay_steps": int(3e5*0.8),
      #"decay_rate": 0.8,
      #"staircase": True},
      #Training MLP on VAE recon
      #Only training MLP weights, not VAE
      #TODO change weight names
      #TODO make option to train only mlp weights in schedule
      {"weights": [
        "mlp/layer0/conv_w_0:0",
        "mlp/layer0/conv_b_0:0",
        "mlp/layer1/conv_w_1:0",
        "mlp/layer1/conv_b_1:0",
        "mlp/layer2/fc_w_2:0",
        "mlp/layer2/fc_b_2:0",
        "mlp/layer3/fc_w_3:0",
        "mlp/layer3/fc_b_3:0"],
      "train_on_adversarial": False,
      "train_vae": False,
      "num_batches": int(1e4),
      "w_decay_mult": 0.0,
      "w_norm_mult": 0.0,
      "kld_mult": 1/self.batch_size,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.8),
      "decay_rate": 0.8,
      "staircase": True},
      ]
    self.schedule = [self.schedule[0].copy()] + self.schedule
    self.schedule[0]["train_on_adversarial"] = False
    self.schedule[0]["num_batches"] = 1000

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.cp_load = True
      self.ae_layer_types = ["fc"]
      self.ae_enc_channels = [768]
      self.ae_activation_functions = ["lrelu", "sigmoid"]
      # MLP params
      self.train_on_recon = True # if False, train on activations
      self.full_data_shape = [28, 28, 1]
      self.num_classes = 10
      self.optimizer = "adam"
      self.mlp_layer_types = ["conv", "conv", "fc", "fc"]
      self.mlp_output_channels = [32, 64, 1024, self.num_classes]
      self.mlp_activation_functions = ["lrelu"]*len(self.mlp_output_channels)
      self.mlp_patch_size = [(5, 5), (5, 5)]
      self.mlp_conv_strides = [(1,1,1,1), (1,1,1,1)]
      self.batch_norm = [None, None, None, None]
      self.mlp_dropout = [1.0, 1.0, 0.4, 1.0]
      self.max_pool = [True, True, False, False]
      self.max_pool_ksize = [(1,2,2,1), (1,2,2,1), None, None]
      self.max_pool_strides = [(1,2,2,1), (1,2,2,1), None, None]
      self.lrn = [None]*len(self.mlp_output_channels)
      # NOTE schedule index will change if vae training is happening
      self.schedule[1]["num_batches"] = int(2e4)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["weight_lr"] = 1e-4
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[1]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.90

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.ae_enc_channels = [768]
      self.ae_dropout = [1.0]*2*len(self.ae_enc_channels)
      self.train_on_recon = True # if False, train on activations
      self.full_data_shape = [16, 16, 1]
      self.num_classes = 2
      self.ae_layer_types = ["fc"]
      self.ae_activation_functions = ["lrelu", "sigmoid"]
      self.mlp_layer_types = ["conv", "fc", "fc"]
      self.mlp_output_channels = [128, 768, self.num_classes]
      self.mlp_activation_functions = ["lrelu"]*len(self.mlp_output_channels)
      self.mlp_patch_size = [(5, 5)]
      self.mlp_conv_strides = [(1,1,1,1)]
      self.batch_norm = [None, None, None]
      self.dropout = [1.0]*len(self.mlp_output_channels)
      self.max_pool = [True, False, False]
      self.max_pool_ksize = [(1,2,2,1), None, None]
      self.max_pool_strides = [(1,2,2,1), None, None]
      self.lrn = [None]*len(self.mlp_output_channels)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.cp_load = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["train_vae"] = True
      self.schedule[sched_idx]["weights"] = None
      self.schedule[sched_idx]["kld_mult"] = 1.0
      self.schedule[sched_idx]["w_decay_mult"] = 0.0
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
