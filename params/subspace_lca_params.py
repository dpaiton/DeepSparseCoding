import os
params = {
  "model_type": "subspace_lca",
  "model_name": "subspace_lca_mnist",
  "version": "0.0",
  "vectorize_data": True,
  "norm_data": False,
  "center_data": True,
  "standardize_data": False,
  "contrast_normalize": False,
  "whiten_data": False,
  "whiten_method": "FT",
  "lpf_data": False, # only for ZCA
  "lpf_cutoff": 0.7,
  "extract_patches": False,
  "batch_size": 100,
  "num_neurons": 512,
  "num_groups": 64,
  "num_steps": 50,
  "dt": 0.001,
  "tau": 0.03,
  "norm_weights": True,
  "optimizer": "annealed_sgd",
  "cp_int": 10000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "log_int": 10,
  "log_to_file": True,
  "gen_plot_int": 2000,
  "save_plots": True,
  "eps": 1e-9,
  "device": "/gpu:0",
  "rand_seed": 123456789,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["phi"],
  "sparse_mult": 1.03,
  "group_orth_mult": 0.1,
  "weight_lr": [0.04],
  "decay_steps": [int(5e4*0.5)],
  "decay_rate": [0.5],
  "staircase": [True],
  "num_batches": int(5e4)}]
