import os
params = {
  "model_type": "ICA",
  "model_name": "test",
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "norm_images": False,
  "norm_weights": True,
  "batch_size": 100,
  "num_pixels": 784,
  "prior": "laplacian",
  "cp_int": 2500,
  "max_cp_to_keep": 5,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_val": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["phi"],
  "log_int": 10,
  "log_to_file": True,
  "gen_plot_int": 100,
  "display_plots": False,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890}

schedule = [
  {"weights": ["a"],
  "weight_lr": [0.08],
  "decay_steps": [2000],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 2000}]
