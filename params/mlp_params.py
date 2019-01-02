import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_a       [bool] If set, l2 normalize layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_pixels   [int] Number of pixels
      num_hidden   [int] Number of layer 1 elements (# hidden units)
      num_classes  [int] Number of layer 2 elements (# categories)
      num_val      [int] Number of validation images
      val_on_cp    [bool] If set, compute validation performance on checkpoint
    """
    super(params, self).__init__()
    self.model_type = "MLP"
    self.model_name = "mlp"
    self.version = "0.0"
    self.optimizer = "annealed_sgd"
    self.vectorize_data = False
    self.rectify_a = True
    self.norm_a = False
    self.norm_weights = True
    self.batch_size = 100
    self.layer_types = ["conv", "fc"]
    self.output_channels = [300, 10]
    self.strides_y = [1, None]
    self.strides_x = [1, None]
    self.patch_size_y = [8, None]
    self.patch_size_x = [8, None]
    self.num_val = 10000
    self.num_labeled = 50000
    self.do_batch_norm = True
    self.cp_int = 100
    self.max_cp_to_keep = 1
    self.val_on_cp = True
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w1"]
    self.log_int = 50
    self.log_to_file = True
    self.gen_plot_int = 1000
    self.save_plots = True

    # If a scalar is provided then this value is broadcast to all trainable variables
    self.schedule = [
      {"num_batches": 1e4,
      "batch_norm_decay_mult": 0.4,
      "weights": None, #["w1", "w2", "bias1", "bias2"],
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.5),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.num_classes = 10
      self.output_channels = [400, 10]
    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_classes = 2
      self.output_channels = [400, 2]
    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.output_channels = [20]+[self.output_channels[-1]]
    self.layer_types = ["conv", "fc"]
    self.patch_size_y = [2, None]
    self.patch_size_x = self.patch_size_y
