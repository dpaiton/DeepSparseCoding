from DeepSparseCoding.tf1x.params.base_params import BaseParams

TRAIN_ON_RECON = False

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of LCA neurons
      num_steps    [int] Number of inference steps
      dt           [float] Discrete global time constant
      tau          [float] LCA time constant
      thresh_type  [str] "hard" or "soft" - LCA threshold function specification
    """
    super(params, self).__init__()
    self.model_type = "mlp_lca_subspace"
    self.train_on_recon = TRAIN_ON_RECON # if False, train on activations
    if(self.train_on_recon):
      self.model_name = "mlp_lca_subspace_recon"
    else:
      self.model_name = "mlp_lca_subspace_latent_1024"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
    self.center_data = False
    self.standardize_data = False
    self.tf_standardize_data = False
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
    # LCA Params
    self.lca_conv = False
    self.num_neurons = 768
    self.num_groups = 192
    self.num_steps = 50
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "sgd"
    # MLP Params
    self.train_on_recon = TRAIN_ON_RECON # if False, train on LCA latent activations
    self.num_val = 10000
    self.num_labeled = 50000
    self.num_classes = 10
    self.mlp_layer_types = ["fc", "fc", "fc"]
    self.mlp_output_channels = [300, 500, self.num_classes]
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
    self.cp_load_name = "lca_subspace_1024_mnist"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["lca_subspace/weights/w:0"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5e3
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "train_lca": False,
      "train_on_adversarial": False,
      "num_batches": int(1e4),
      "sparse_mult": 0.01,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.8),
      "decay_rate": 0.8,
      "staircase": True},
      ]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.cp_int = 1e4
      self.gen_plot_int = 1e4
      # LCA params
      self.lca_conv = False
      self.num_neurons = 512
      self.num_groups = 128
      if self.train_on_recon:
        self.full_data_shape = [28, 28, 1]
        self.num_classes = 10
        self.optimizer = "adam"
        self.mlp_layer_types = ["conv", "conv", "fc", "fc"]
        self.mlp_activation_functions = ["relu", "relu", "relu", "identity"]
        self.mlp_output_channels = [32, 64, 1024, self.num_classes]
        self.mlp_patch_size = [(5, 5), (5, 5)]
        self.mlp_conv_strides = [(1,1,1,1), (1,1,1,1)]
        self.batch_norm = [None, None, None, None]
        self.mlp_dropout = [1.0, 1.0, 0.4, 1.0]
        self.max_pool = [True, True, False, False]
        self.max_pool_ksize = [(1,2,2,1), (1,2,2,1), None, None]
        self.max_pool_strides = [(1,2,2,1), (1,2,2,1), None, None]
        self.lrn = [None]*len(self.mlp_output_channels)
        # NOTE schedule index will change if lca training is happening
        self.schedule[-1]["num_batches"] = int(1e5)
        for sched_idx in range(len(self.schedule)):
          self.schedule[sched_idx]["weights"] = [
            "mlp/layer0/conv_w_0:0",
            "mlp/layer0/conv_b_0:0",
            "mlp/layer1/conv_w_1:0",
            "mlp/layer1/conv_b_1:0",
            "mlp/layer2/fc_w_2:0",
            "mlp/layer2/fc_b_2:0",
            "mlp/layer3/fc_w_3:0",
            "mlp/layer3/fc_b_3:0"]
          self.schedule[sched_idx]["train_on_adversarial"] = False
          self.schedule[sched_idx]["group_orth_mult"] = 0.04
          self.schedule[sched_idx]["sparse_mult"] = 0.45
          self.schedule[sched_idx]["weight_lr"] = 1e-4
          self.schedule[sched_idx]["decay_steps"] = int(0.5*self.schedule[sched_idx]["num_batches"])
          self.schedule[sched_idx]["decay_rate"] = 0.9
      else: #96.34 val 98.00 train
        self.mlp_output_channels = [128, self.num_classes]
        self.mlp_layer_types = ["fc", "fc"]
        self.mlp_activation_functions = ["lrelu", "identity"]
        self.optimizer = "adam"
        self.mlp_patch_size = []
        self.mlp_conv_strides = []
        # TODO: tune decay & norm mults
        self.mlp_decay_mult = 0
        self.mlp_norm_mult = 1e-4
        self.batch_norm = [None]*len(self.mlp_output_channels)
        self.mlp_dropout = [0.8, 1.0]
        self.max_pool = [False, False]
        self.max_pool_ksize = [None, None]
        self.max_pool_strides = [None, None]
        self.lrn = [None]*len(self.mlp_output_channels)
        for sched_idx in range(len(self.schedule)):
          self.schedule[sched_idx]["weights"] = [
            "mlp/layer0/fc_w_0:0",
            "mlp/layer0/fc_b_0:0",
            "mlp/layer1/fc_w_1:0",
            "mlp/layer1/fc_b_1:0"]
          self.schedule[sched_idx]["train_on_adversarial"] = False
          self.schedule[sched_idx]["num_batches"] = int(2e5)
          self.schedule[sched_idx]["group_orth_mult"] = 0.04
          self.schedule[sched_idx]["sparse_mult"] = 0.45
          self.schedule[sched_idx]["weight_lr"] = 1e-5
          self.schedule[sched_idx]["decay_steps"] = int(0.4*self.schedule[sched_idx]["num_batches"])
          self.schedule[sched_idx]["decay_rate"] = 0.9

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
      self.num_groups = 192
      self.train_on_recon = True # if False, train on activations
      self.num_classes = 2
      self.mlp_output_channels = [128, 64, self.num_classes]
      self.mlp_activation_functions = ["lrelu"]*len(self.mlp_output_channels)
      self.lrn = [None]*len(self.mlp_output_channels)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["group_orth_mult"] = 0.04
        self.schedule[sched_idx]["sparse_mult"] = 0.21
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    self.cp_load = False
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["weights"] = None
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
    self.num_groups = 25
    self.num_steps = 5
