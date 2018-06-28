import os
import numpy as np

params = {
  "model_type": "conv_gdn_decoder",
  "model_name": "conv_gdn_decoder",
  "version": "0.0",
  "input_shape": [8,8,7],
  "batch_size": 1,
  "input_channels": [7, 128, 128],
  "output_channels": [128, 128, 1],
  "patch_size_y": [5, 5, 9],
  "patch_size_x": [5, 5, 9],
  "strides": [2,2,4],
  "w_thresh_min": 1e-3,
  "b_thresh_min": 1e-3,
  "gdn_mult_min": 1e-6,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/",
  "data_file":"/media/tbell/datasets/verified_images.txt"}
