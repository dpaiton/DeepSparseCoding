import os
from params.base_params import BaseParams

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
    self.model_name = "mlp"
    self.version = "0.0"
    self.optimizer = "annealed_sgd"
    self.vectorize_data = False
    self.batch_size = 100
    self.num_classes = 10
    self.layer_types = ["conv", "fc"]
    self.output_channels = [300, self.num_classes]
    self.patch_size_y = [8, None]
    self.patch_size_x = [8, None]
    self.conv_strides = [(1,1,1,1), None]
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
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w1"] #None means load everything
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True

    # If a scalar is provided then this value is broadcast to all trainable variables
    self.schedule = [
      {"num_batches": 1e4,
      "weights": None,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.5),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = False
      self.rescale_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      self.num_classes = 10
      self.optimizer = "adam"
      self.layer_types = ["conv", "conv", "fc", "fc"]
      self.output_channels = [32, 64, 1024, self.num_classes]
      self.patch_size_y = [5, 5, None, None]
      self.patch_size_x = self.patch_size_y
      self.conv_strides = [(1,1,1,1), (1,1,1,1), None, None]
      self.batch_norm = [None, None, None, None]
      self.dropout = [1.0, 1.0, 0.4, 1.0] # TODO: Set dropout defaults somewhere
      self.max_pool = [True, True, False, False]
      self.max_pool_ksize = [(1,2,2,1), (1,2,2,1), None, None]
      self.max_pool_strides = [(1,2,2,1), (1,2,2,1), None, None]
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(2e4)
        self.schedule[sched_idx]["weight_lr"] = 1e-4
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.90

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_classes = 2
      self.output_channels[-1] = self.num_classes

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
    self.output_channels = [20]+[self.output_channels[-1]]
    self.layer_types = ["conv", "fc"]
    self.patch_size_y = [2, None]
    self.patch_size_x = self.patch_size_y
