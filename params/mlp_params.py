import os
from params.base_params import BaseParams

TRAIN_ADV = True

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      batch_size   [int] Number of images in a training batch
      num_classes  [int] Number of categories
      num_val      [int] Number of validation images
      val_on_cp    [bool] If set, compute validation performance on checkpoint
    """
    super(params, self).__init__()
    self.model_type = "mlp"
    if(TRAIN_ADV):
      self.model_name = "mlp_adv"
    else:
      self.model_name = "mlp_768"
    self.version = "0.0"
    self.optimizer = "annealed_sgd"
    self.vectorize_data = False
    self.standardize_data = False
    self.tf_standardize_data = False
    self.batch_size = 100
    self.num_classes = 10
    self.mlp_layer_types = ["conv", "fc"]
    self.mlp_activation_functions = ["relu", "identity"]
    self.mlp_output_channels = [300, self.num_classes]
    self.mlp_patch_size = [(8, 8)]
    self.mlp_conv_strides = [(1,1,1,1)]
    self.num_val = 10000
    self.num_labeled = 50000
    self.batch_norm = [0.4, None]
    self.dropout = [1.0, 1.0]
    self.max_pool = [False, False]
    self.max_pool_ksize = [None, None]
    self.max_pool_strides = [None, None]
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.val_on_cp = True
    self.eval_batch_size = 1000
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w1"] #None means load everything
    self.log_to_file = True
    self.gen_plot_int = 1e3
    self.save_plots = True
    self.mlp_decay_mult = 0 # TODO: make an auto-placeholder
    self.mlp_norm_mult = 1e-4 # TODO: make an auto-placeholder
    #Adversarial params
    self.adversarial_num_steps = 40
    self.adversarial_attack_method = "kurakin_untargeted"
    self.adversarial_step_size = 0.01
    self.adversarial_max_change = 0.3
    self.adversarial_target_method = "random" #Not used if attach_method is untargeted
    self.adversarial_clip = True
    #TODO get these params from other params
    self.adversarial_clip_range = [0.0, 1.0]
    #Tradeoff in carlini attack between input pert and target
    self.carlini_recon_mult = 1

    # If a scalar is provided then this value is broadcast to all trainable variables
    self.schedule = [
      {"num_batches": int(1e4),
      "train_on_adversarial": False,
      "weights": None,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.5),
      "decay_rate": 0.8,
      "staircase": True}]
    if(TRAIN_ADV):
      self.schedule = [self.schedule[0].copy()] + self.schedule
      self.schedule[0]["train_on_adversarial"] = False
      self.schedule[1]["train_on_adversarial"] = True
      self.schedule[0]["num_batches"] = 1000

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = False
      self.standardize_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.log_int = 100
      self.cp_int = 1e4
      self.gen_plot_int = 2e4
      self.num_classes = 10
      self.optimizer = "adam"
      self.mlp_layer_types = ["fc", "fc", "fc"]
      self.mlp_activation_functions = ["lrelu", "lrelu", "identity"]
      self.mlp_output_channels = [768, 512, self.num_classes]
      self.mlp_patch_size = []
      self.mlp_conv_strides = []
      self.batch_norm = [None, None, None]
      self.dropout = [0.2, 0.4, 1.0] # TODO: Set dropout defaults somewhere
      self.lrn = [None, None, None]
      self.max_pool = [False, False, False]
      self.max_pool_ksize = [None, None, None]
      self.max_pool_strides = [None, None, None]
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(5e5)
        self.schedule[sched_idx]["weight_lr"] = 1e-4
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.90
      if(TRAIN_ADV):
        self.schedule[0]["num_batches"] = int(1e4)

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.vectorize_data = False
      self.rescale_data = False
      self.standardize_data = False
      self.tf_standardize_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.log_int = 100
      self.cp_int = 500
      self.gen_plot_int = 1e3
      self.num_classes = 10
      self.optimizer = "adam"
      self.mlp_layer_types = ["conv", "conv", "fc", "fc", "fc"]
      self.mlp_activation_functions = ["lrelu", "lrelu", "lrelu", "lrelu", "identity"]
      self.mlp_output_channels = [256, 64, 384, 192, self.num_classes]
      #TF model does lrn after pool in conv1, lrn before pool in conv2
      #TODO test if this matters
      #String can be post or pre, depending on applying LRN before or after pooling
      self.lrn = [None, None, None, None, None]
      self.mlp_patch_size = [(12, 12), (5, 5)]
      self.mlp_conv_strides = [(1,2,2,1), (1,1,1,1)]
      self.batch_norm = [None, None, None, None, None]
      self.dropout = [0.5, 0.5, 0.5, 0.5, 1.0] # TODO: Set dropout defaults somewhere
      self.max_pool = [False, True, False, False, False]
      self.max_pool_ksize = [None, (1,3,3,1), None, None, None]
      self.max_pool_strides = [None, (1,2,2,1), None, None, None]
      self.batch_size = 128

      self.adversarial_num_steps = 10
      self.adversarial_step_size = 0.01
      self.adversarial_max_change = 0.03

      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["weight_lr"] = 5e-4
        #Decay steps is in terms of epochs, (num_epochs_per_batch * 350 per decay)
        self.schedule[sched_idx]["decay_steps"] = 80000
        self.schedule[sched_idx]["decay_rate"] = 0.9
      if(TRAIN_ADV):
        self.schedule[0]["num_batches"] = int(5e3)
        self.schedule[1]["num_batches"] = int(3e5)

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_classes = 2
      self.mlp_output_channels[-1] = self.num_classes

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 50
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.mlp_output_channels = [20]+[self.mlp_output_channels[-1]]
    self.mlp_layer_types = ["conv", "fc"]
    self.mlp_activation_functions = ["lrelu"]*len(self.mlp_output_channels)
    self.mlp_patch_size = [(2, 2)]
    self.mlp_conv_strides = [(1,1,1,1)]
    self.batch_norm = [None, None]
    self.dropout = [1.0, 1.0]
    self.lrn = ["post", None]
    self.max_pool = [True, False]
    self.max_pool_ksize = [(1,2,2,1), None]
    self.max_pool_strides = [(1,2,2,1), None]
