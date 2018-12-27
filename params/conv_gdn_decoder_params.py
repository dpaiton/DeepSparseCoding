import os
import numpy as np
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "conv_gdn_decoder"
    self.model_name = "conv_gdn_decoder"
    self.version = "0.0"
    self.input_shape = [4, 4, 28]
    self.batch_size = 1
    self.input_channels = [28, 64, 128]
    self.output_channels = [64, 128, 1]
    self.patch_size_y = [4, 9, 8]
    self.patch_size_x = [4, 9, 8]
    self.strides = [2, 2, 4]
    self.gdn_w_init_const = 0.1
    self.gdn_b_init_const = 0.1
    self.gdn_w_thresh_min = 1e-3
    self.gdn_b_thresh_min = 1e-3
    self.gdn_eps = 1e-6
    self.data_file ="/media/tbell/datasets/verified_images.txt"
