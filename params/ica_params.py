import os
params = {
  "model_type": "ica",
  "model_name": "ica_nowhite",
  "version": "0.0",
  "num_images": 50,
  "vectorize_images": True,
  "norm_images": False,
  "whiten_images": False,
  "extract_patches": True,
  "num_patches": 1e6,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 100,
  #"prior": "cauchy",
  "prior": "laplacian",
  "optimizer": "annealed_sgd",
  "cp_int": 10000,
  "max_cp_to_keep": 5,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_step": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["a"],
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 500,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["a"],
  "weight_lr": [0.01],
  "decay_steps": [30000],
  "decay_rate": [0.6],
  "staircase": [True],
  "num_batches": 30000}]
