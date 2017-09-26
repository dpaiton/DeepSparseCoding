import os
params = {
  "model_type": "lca_pca_fb",
  "model_name": "lca_pca_fb",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "rectify_a": True,
  "norm_weights": True,
  "whiten_images": True,
  "contrast_normalize": False,
  "epoch_size": 1e6,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "patch_variance_threshold": 1e-6,
  "batch_size": 100,
  "num_neurons": 512,
  "num_pooling_units": 25,
  "num_steps": 30,
  "dt": 0.001,
  "tau": 0.03,
  "thresh_type": "soft",
  "cp_int": 1000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "log_int": 10,
  "log_to_file": True,
  "gen_plot_int": 10,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 12345,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}

schedule = [
  {"weights": ["phi"],
  "sparse_mult": 0.4,
  "fb_mult": 0.0,
  "weight_lr": [0.07],
  "decay_steps": [8000],
  "decay_rate": [0.5],
  "num_batches": 25000}]
