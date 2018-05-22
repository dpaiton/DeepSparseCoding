import os
params = {
  "model_type": "ica",
  "model_name": "ica",
  "version": "0.0",
  "num_images": 100,
  "vectorize_data": True,
  "norm_data": False,
  "center_data": False,
  "standardize_data": False,
  "contrast_normalize": False,
  "lpf_data": False, # only for ZCA/PCA
  "lpf_cutoff": 0.7,
  "whiten_data": True,
  "whiten_method": "FT",
  "extract_patches": True,
  "num_patches": 1e6,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 0,
  "batch_size": 50,
  "prior": "laplacian", #"cauchy",
  "optimizer": "annealed_sgd",
  "cp_int": 10000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_step": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["a"],
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 1000,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["w_synth"], # A, VS265
  #{"weights": ["w_analysis"], # W=A^-1, Bell & Sejnowski
  "weight_lr": [0.001],
  "decay_steps": [3e5],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 5e5}]
