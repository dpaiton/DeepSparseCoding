import os
import numpy as np

params = {
  "model_type": "conv_gdn_decoder",
  "model_name": "conv_gdn_decoder",
  "version": "0.0",
  "input_shape": [4, 4, 28],
  "batch_size": 1,
  "input_channels": [28, 64, 128],
  "output_channels": [64, 128, 1],
  "patch_size_y": [4, 8, 9],
  "patch_size_x": [4, 8, 9],
  "strides": [2, 2, 4],
  "gdn_w_init_const": 0.1,
  "gdn_b_init_const": 0.1,
  "gdn_w_thresh_min": 1e-6,
  "gdn_b_thresh_min": 1e-6,
  "gdn_eps": 1e-6,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/",
  "data_file":"/media/tbell/datasets/verified_images.txt"}
