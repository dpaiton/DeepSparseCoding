import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      stride_x      [int] convolutional stride (x)
      stride_y      [int] convolutional stride (y)
      patch_size_y  [int] the number of rows in the patch
      patch_size_x  [int] the number of columns in the patch
    """
    super(params, self).__init__()
    self.model_type = "lca_conv"
    self.model_name = "lca_conv"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = False
    self.norm_data = False
    self.center_data = True
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False # only for ZCA
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.image_edge_size = 128
    self.batch_size = 25
    self.num_neurons = 128 # pixel-overcompleteness is num_neurons/(stride_y * stride_x)
    self.stride_y = 8
    self.stride_x = 8
    self.patch_size_y = 16 # weight receptive field
    self.patch_size_x = 16
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "annealed_sgd"
    self.cp_int = 5000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.log_int = 10
    self.log_to_file = True
    self.gen_plot_int = 1000
    self.save_plots = True
    self.schedule = [
      {"weights": ["lca_conv/weights/w:0"],
      "num_batches": int(1e5),
      "sparse_mult": 0.4,
      "weight_lr": [0.001],
      "decay_steps": [int(1e5*0.8)],
      "decay_rate": [0.8],
      "staircase": [True]}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.rescale_data = True
      self.center_data = False
      self.whiten_data = False
      self.lpf_data = False # only for ZCA
      self.lpf_cutoff = 0.7
      self.num_neurons = 768
      self.stride_y = 2
      self.stride_x = 2
      self.patch_size_y = 8 # weight receptive field
      self.patch_size_x = 8
      for schedule_idx in range(len(self.schedule)):
        self.schedule[schedule_idx]["sparse_mult"] = 0.21
        self.schedule[schedule_idx]["weight_lr"] = [0.1]

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 150
      self.rescale_data = False
      self.center_data = True
      self.whiten_data = True
      self.whiten_method = "FT"
      self.lpf_data = False # FT whitening already does LPF
      self.lpf_cutoff = 0.7
      self.image_edge_size = 128
      self.stride_y = 8
      self.stride_x = 8
      self.patch_size_y = 16 # weight receptive field
      self.patch_size_x = 16
      self.num_neurons = 128
      self.num_steps = 60
      self.dt = 0.001
      self.tau = 0.03
      self.rectify_a = True
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.8
        self.schedule[sched_idx]["weight_lr"] = [0.001]
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = [int(0.8*self.schedule[sched_idx]["num_batches"])]

    elif data_type.lower() == "field":
      self.model_name += "_field"
      self.batch_size = 1
      self.rescale_data = False
      self.center_data = True
      self.whiten_data = True
      self.whiten_method = "FT"
      self.lpf_data = False # FT whitening already does LPF
      self.lpf_cutoff = 0.7
      self.image_edge_size = 128
      self.stride_y = 8
      self.stride_x = 8
      self.patch_size_y = 16 # weight receptive field
      self.patch_size_x = 16
      self.num_neurons = 128
      self.num_steps = 60
      self.dt = 0.001
      self.tau = 0.03
      self.rectify_a = True
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.8
        self.schedule[sched_idx]["weight_lr"] = [0.001]
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = [int(0.8*self.schedule[sched_idx]["num_batches"])]

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.batch_size = 12
      self.standardize_data = True
      self.rescale_data = False
      self.center_data = False
      self.whiten_data = False # True
      self.whiten_method = "FT"
      self.lpf_data = False # FT whitening already does LPF
      self.lpf_cutoff = 0.7
      self.image_edge_size = 128
      self.stride_y = 2
      self.stride_x = 2
      self.patch_size_y = 12 #8 # weight receptive field
      self.patch_size_x = 12 #8
      self.num_neurons = 256 # 128
      self.num_steps = 100
      self.dt = 0.001
      self.tau = 0.1
      self.rectify_a = True
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.07 #0.1
        self.schedule[sched_idx]["weight_lr"] = [0.001]
        self.schedule[sched_idx]["num_batches"] = int(1e6)
        self.schedule[sched_idx]["decay_steps"] = [int(0.8*self.schedule[sched_idx]["num_batches"])]

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 8
      self.rescale_data = True
      self.center_data = True
      self.whiten_data = False
      self.lpf_data = False # only for ZCA
      self.lpf_cutoff = 0.7
      self.num_neurons = 32
      self.stride_y = 2
      self.stride_x = 2
      self.patch_size_y = 8 # weight receptive field
      self.patch_size_x = 8
      for schedule_idx in range(len(self.schedule)):
        self.schedule[schedule_idx]["sparse_mult"] = 0.21
        self.schedule[schedule_idx]["weight_lr"] = [0.1]

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = [1e-4]
    self.num_neurons = 100
    self.stride_y = 1
    self.stride_x = 1
    self.patch_size_y = 2
    self.patch_size_x = 2
