import os
import numpy as np
params = {
  "model_type": "ica_pca",
  "model_name": "ica_pca",
  "version": "0.0",
  "num_images": 50,
  "vectorize_data": True,
  "norm_data": False,
  "whiten_data": True,
  "contrast_normalize": False,
  "extract_patches": True,
  "num_patches": 1e5,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 100,
  #"prior": "cauchy",
  "prior": "laplacian",
  "optimizer": "annealed_sgd",
  "num_pooling_units": 50,
  "cp_int": 10000,
  "max_cp_to_keep": 2,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_step": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["a"],
  "log_int": 500,
  "log_to_file": True,
  "gen_plot_int": 500,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 12345,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["a"],
  "weight_lr": [0.01],
  "decay_steps": [int(np.floor(1e5*0.8))],
  "decay_rate": [0.7],
  "staircase": [True],
  "num_batches": int(1e5)}]
