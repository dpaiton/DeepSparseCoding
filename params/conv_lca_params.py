import os
params = {
  "model_type": "conv_lca",
  "model_name": "conv_lca_cifar",
  "version": "0.0",
  "center_data": False,
  "norm_data": False,
  "whiten_data": True,
  "whiten_method": "FT",
  "standardize_data": False,
  "contrast_normalize": False,
  "extract_patches": True,
  "num_patches": 1e6,
  "patch_edge_size": 32, # image patches
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 10,
  "stride_y": 2,
  "stride_x": 2,
  "patch_size_y": 12, # weight receptive field
  "patch_size_x": 12,
  "num_neurons": 96, # pixel-overcompleteness is num_neurons/(stride_y * stride_x)
  "num_steps": 120,
  "dt": 0.001,
  "tau": 0.02,
  "rectify_a": True,
  "norm_weights": True,
  "thresh_type": "soft",
  "optimizer": "annealed_sgd",
  "cp_int": 1000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "log_int": 50,
  "log_to_file": True,
  "gen_plot_int": 500,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["phi"],
  "sparse_mult": 0.01,
  "weight_lr": [0.001],
  "decay_steps": [int(1e5*0.8)],
  "decay_rate": [0.6],
  "staircase": [True],
  "num_batches": int(1e5)}]
