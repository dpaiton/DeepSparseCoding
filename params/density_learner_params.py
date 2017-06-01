import os
params = {
  "model_type": "density_learner",
  "model_name": "density",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "rectify_a": True,
  "rectify_v": True,
  "norm_weights": True,
  "whiten_images": True,
  "epoch_size": 1e4,
  "patch_edge_size": 20,
  "overlapping_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 1000,
  "num_neurons": 900,
  "num_v": 100,
  "num_u_steps": 20,
  "num_v_steps": 100,
  "dt": 0.001,
  "tau": 0.03,
  "v_step_scale": 0.01,
  "cp_int": 20000,
  "max_cp_to_keep": 5,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_step": 120000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["phi"],
  "log_to_file": True,
  "log_int": 10,
  "gen_plot_int": 100,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["b"],
  "u_fb_mult": 0.12,
  "v_sparse_mult": 0.2,
  "b_decay_mult": 0.0,
  "weight_lr": [0.001],
  "decay_steps": [40000],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 40000}]
