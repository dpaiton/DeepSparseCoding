import os
params = {
  "model_type": "rica",
  "model_name": "rica_bfgs",
  "version": "0.0",
  "num_images": 150,
  "vectorize_data": True,
  "norm_data": False,
  "center_data": False,
  "standardize_data": True,
  "contrast_normalize": False,
  "whiten_data": True,
  "whiten_method": "FT",
  "lpf_data": False,
  "lpf_cutoff": 0.7,
  "extract_patches": True,
  "num_patches": 1e6,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "randomize_patches": True,
  "patch_variance_threshold": 0.0,
  "batch_size": 50,
  "num_neurons": 512,
  "norm_weights": False,
  "optimizer": "lbfgsb",#"adam",#"annealed_sgd",
  "maxiter": 100,
  "cp_int": 100000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "cp_load_name": "rica_pretrain", # trained with sgd
  "cp_load_step": None, # latest checkpoint
  "cp_load_ver": "0.0",
  "cp_load_var": ["w"],
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 100,
  "save_plots": True,
  "eps": 1e-7,
  "device": "/gpu:0",
  "rand_seed": 123456789,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["w"],
  "recon_mult": 7.0,
  "sparse_mult": 1.0,
  "weight_lr": [0.05],
  "decay_steps": [int(1e8*0.9)],
  "decay_rate": [0.5],
  "staircase": [True],
  "num_batches": int(1e8)}]