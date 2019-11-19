import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of LISTA neurons
      num_steps    [int] Number of inference steps
      dt           [float] Discrete global time constant
      thresh_type  [str] "hard" or "soft" - LISTA threshold function specification
    """
    super(params, self).__init__()
    self.model_type = "mlp_lista"
    self.model_name = "mlp_lista_5_adv"
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
    self.optimizer = "annealed_sgd"
    # LCA Params
    self.num_steps = 50
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = False
    # LISTA Params
    self.num_layers = 5
    self.num_neurons = 768
    self.thresh_type = "soft"
    # MLP Params
    self.num_val = 10000
    self.num_labeled = 50000
    self.num_classes = 10
    self.mlp_output_channels = [300, 500, self.num_classes]
    num_mlp_layers = len(self.mlp_output_channels)
    self.mlp_activation_functions = ["relu"]*len(self.mlp_output_channels)
    self.mlp_layer_types = ["fc"]*num_mlp_layers
    self.mlp_patch_size = []
    self.mlp_conv_strides = []
    self.batch_norm = [None]*num_mlp_layers
    self.mlp_dropout = [1.0]*num_mlp_layers
    self.max_pool = [False]*num_mlp_layers
    self.max_pool_ksize = [None]*num_mlp_layers
    self.max_pool_strides = [None]*num_mlp_layers
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
    self.cp_int = 1e4
    self.val_on_cp = True
    self.eval_batch_size = 100
    self.max_cp_to_keep = 1
    self.cp_load = True
    self.cp_load_name = "lista_5_mnist"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["weights/w_enc:0", "weights/lateral_connectivity:0"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 1e4
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "train_on_adversarial":True,
      "num_batches": int(1e4),
      "sparse_mult": 0.01,
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
      self.whiten_data = False
      self.extract_patches = False
      self.log_int = 100
      self.cp_int = 1e4
      self.gen_plot_int = 1e4
      # LISTA params
      self.num_layers = 5
      self.num_neurons = 768
      # MLP params
      self.mlp_output_channels = [1536, 1200, self.num_classes]
      self.mlp_layer_types = ["fc"]*len(self.mlp_output_channels)
      self.mlp_activation_functions = ["relu"]*len(self.mlp_output_channels)
      self.optimizer = "adam"
      self.mlp_patch_size = []
      self.mlp_conv_strides = []
      self.batch_norm = [None]*len(self.mlp_output_channels)
      self.mlp_dropout = [0.5, 0.4, 1.0]
      self.max_pool = [False]*len(self.mlp_output_channels)
      self.max_pool_ksize = [None]*len(self.mlp_output_channels)
      self.max_pool_strides = [None]*len(self.mlp_output_channels)
      self.lrn = [None]*len(self.mlp_output_channels)
      self.schedule[1]["num_batches"] = int(4e4)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.19
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
      self.num_neurons = 768
      self.train_on_recon = True # if False, train on activations
      self.num_classes = 2
      self.mlp_output_channels = [128, 64, self.num_classes]
      self.mlp_activation_functions = ["relu"]*len(self.mlp_output_channels)
      self.lrn = [None]*len(self.mlp_output_channels)
      for sched_idx in range(len(self.schedule)):
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
    self.num_steps = 5
