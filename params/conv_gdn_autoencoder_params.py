import os
import numpy as np

params = {
  "model_type": "conv_gdn_autoencoder",
  "model_name": "conv_gdn_autoencoder_pretrain_exp",
  "version": "0.0",
  "vectorize_data": False,
  "norm_data": False,
  "center_data": False,
  "standardize_data": False,
  "contrast_normalize": False,
  "whiten_data": True,
  "whiten_method": "FT",
  "lpf_data": True,
  "lpf_cutoff": 0.7,
  "extract_patches": False,
  "im_size_y": 64,
  "im_size_x": 64,
  "batch_size": 50,
  "num_colors": 1,
  "downsample_images": True,
  "downsample_method": "resize",
  "num_preproc_threads": 8,
  "mem_v_min": -1.0,
  "mem_v_max": 1.0,
  "sigmoid_beta": 1.0,
  "mle_step_size": 0.01,
  "num_mle_steps": 15,
  "num_triangles": 20,
  "n_mem": 448,
  "input_channels": [1, 128, 64],
  "output_channels": [128, 64, 28],
  "patch_size_y": [8, 9, 4],
  "patch_size_x": [8, 9, 4],
  "strides": [4, 2, 2],
  "w_thresh_min": 1e-3,
  "b_thresh_min": 1e-3,
  "gdn_mult_min": 1e-5,
  "memristor_type": "rram",
  "memristor_data_loc": os.path.expanduser("~")+"/CAE_Project/CAEs/data/Partial_Reset_PCM.pkl",
  "optimizer": "annealed_sgd",
  "cp_int": 10000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "cp_load_name": "conv_gdn_autoencoder_pretrain",
  "cp_load_step": 10000,
  "cp_load_ver": "1.0",
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 10000,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/gpu:0",
  "rand_seed": 1234567890,
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/",
  "data_file":"/media/tbell/datasets/verified_images.txt"}

w_list = ["w"+str(idx) for idx in range(2*len(params["input_channels"]))]
b_list = ["b"+str(idx) for idx in range(2*len(params["input_channels"]))]
w_gdn_list = ["w_gdn"+str(idx) for idx in range(len(params["input_channels"]))]
b_gdn_list = ["b_gdn"+str(idx) for idx in range(len(params["input_channels"]))]
# Don't do igdn on last layer (reconstruction)
w_igdn_list = ["w_igdn"+str(idx)
  for idx in range(len(params["input_channels"]), 2*len(params["input_channels"])-1)]
b_igdn_list = ["b_igdn"+str(idx)
  for idx in range(len(params["input_channels"]), 2*len(params["input_channels"])-1)]

conv_list = w_list + b_list
gdn_list = w_gdn_list + b_gdn_list + w_igdn_list + b_igdn_list
train_list = conv_list +  gdn_list
params["cp_load_var"] = train_list

weight_lr = [3.0e-4 for _ in range(len(conv_list))]
weight_lr += [1.0e-4 for _ in range(len(gdn_list))]
decay_rate = [0.8 for _ in range(len(train_list))]
staircase = [True for _ in range(len(train_list))]

schedule = [
  {"weights": train_list,
  "ent_mult": 0.1,
  "ramp_slope": 0.8,
  "decay_mult": 0.002,
  "noise_var_mult": 0.0,
  "mem_error_rate": 0.0,
  "triangle_centers": np.linspace(-1.0, 1.0, params["num_triangles"]),
  "weight_lr": weight_lr,
  "num_epochs": 10, #10 takes about 13 hours on rw1 with batch size 50
  "decay_rate": decay_rate,
  "staircase": staircase}]
