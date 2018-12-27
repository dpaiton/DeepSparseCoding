import os
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      stride_x      [int] convolutional stride (x)
      stride_y      [int] convolutional stride (y)
      patch_size_y  [int] the number of rows in the patch
      patch_size_x  [int] the number of columns in the patch
    """
    super(params, self).__init__()
    self.model_type = "conv_lca"
    self.model_name = "conv_lca"
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
    self.num_batches = int(1e5)
    self.batch_size = 25
    self.stride_y = 8
    self.stride_x = 8
    self.patch_size_y = 16 # weight receptive field
    self.patch_size_x = 16
    self.num_neurons = 128 # pixel-overcompleteness is num_neurons/(stride_y * stride_x)
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.log_int = 10
    self.log_to_file = True
    self.gen_plot_int = 2000
    self.save_plots = True

    self.schedule = [
      {"weights": ["phi"],
      "sparse_mult": 0.4,
      "weight_lr": [0.001],
      "decay_steps": [int(self.num_batches*0.8)],
      "decay_rate": [0.8],
      "staircase": [True]}]
