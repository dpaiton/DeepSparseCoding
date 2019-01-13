import os
from params.base_params import BaseParams

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
    self.num_images = 150
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
    self.vae_output_channels = [512, 50]
    self.latent_act_func = "relu"
    self.noise_level = 0.0 # variance of noise added to the input data
    self.optimizer = "adam"
    # MLP Params
    self.train_on_recon = True # if False, train on LCA latent activations
    self.num_classes = 10
    self.layer_types = ["fc", "fc", "fc"]
    self.mlp_output_channels = [300, 500, self.num_classes]
    self.patch_size_y = [None, None, None]
    self.patch_size_x = [None, None, None]
    self.conv_strides = [None, None, None]
    self.batch_norm = [None, None, None]
    self.dropout = [None, None, None]
    self.max_pool = [False, False, False]
    self.max_pool_ksize = [None, None, None]
    self.max_pool_strides = [None, None, None]
    # Others
    self.cp_int = 10000
    self.val_on_cp = True
    self.max_cp_to_keep = None
    self.cp_load = True
    self.cp_load_name = "vae_two_layer_mnist"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = [
        "w_enc_2_std:0",
        "b_enc_2_std:0",
        "layer0/w_0:0",
        "layer0/b_0:0",
        "layer1/w_1:0",
        "layer1/b_1:0",
        "layer3/w_3:0",
        "layer3/b_3:0",
        "layer4/w_4:0",
        "layer4/b_4:0",
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
      #"sparse_mult": 0.0,
      #"decay_mult": 0.0,
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
        "layer0/conv_w_0:0",
        "layer0/conv_b_0:0",
        "layer1/conv_w_1:0",
        "layer1/conv_b_1:0",
        "layer2/fc_w_2:0",
        "layer2/fc_b_2:0",
        "layer3/fc_w_3:0",
        "layer3/fc_b_3:0"
        ],
      "train_vae": False,
      "num_batches": int(1e4),
      "sparse_mult": 0.01,
      "decay_mult": 0.0,
      "kld_mult": 1/self.batch_size,
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
      # MLP params
      self.train_on_recon = True # if False, train on activations
      self.full_data_shape = [28, 28, 1]
      self.num_classes = 10
      self.optimizer = "adam"
      self.layer_types = ["conv", "conv", "fc", "fc"]
      self.mlp_output_channels = [32, 64, 1024, self.num_classes]
      self.patch_size_y = [5, 5, None, None]
      self.patch_size_x = self.patch_size_y
      self.conv_strides = [(1,1,1,1), (1,1,1,1), None, None]
      self.batch_norm = [None, None, None, None]
      self.dropout = [None, None, 0.4, None]
      self.max_pool = [True, True, False, False]
      self.max_pool_ksize = [(1,2,2,1), (1,2,2,1), None, None]
      self.max_pool_strides = [(1,2,2,1), (1,2,2,1), None, None]
      # NOTE schedule index will change if lca training is happening
      self.schedule[0]["num_batches"] = int(2e4)
      self.schedule[0]["sparse_mult"] = 0.21
      self.schedule[0]["weight_lr"] = 1e-4
      self.schedule[0]["decay_steps"] = int(0.8*self.schedule[0]["num_batches"])
      self.schedule[0]["decay_rate"] = 0.90

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.train_on_recon = True # if False, train on activations
      self.num_classes = 2
      self.mlp_output_channels = [128, 64, self.num_classes]
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
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["weights"] = None
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
    self.num_steps = 5
