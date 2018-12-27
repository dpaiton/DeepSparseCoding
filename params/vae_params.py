import os
params = {
  "model_type": "vae",
  "model_name": "vae_mnist_test",
  "version": "0.0",
  "vectorize_data": True,
  "norm_data": False,
  "rescale_data": True,
  "center_data": False,
  "standardize_data": False,
  "contrast_normalize": False,
  "whiten_data": False,
  "lpf_data": False,
  "lpf_cutoff": 0.7,
  "extract_patches": False,
  "batch_size": 200,
  #Specify number of neurons for encoder
  #Last element in list is the size of the latent space
  #Decoder will automatically build the transpose of the encoder
  "num_neurons": [512, 50],
  "optimizer": "adam",
  "cp_int": 10000,
  "max_cp_to_keep": 1,
  "cp_load": False,
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 10000,
  "save_plots": True,
  "eps": 1e-6,
  "device": "/gpu:0",
  "rand_seed": 123456789,
  "out_dir": os.path.expanduser("~")+"/mountData/",
  "data_dir": os.path.expanduser("~")+"/mountData/datasets/",
  }

#decoders mirror encoder layers
num_encoder_layers = len(params["num_neurons"])
encoder_layer_range = list(range(num_encoder_layers))
w_enc_list = ["w_enc_" + str(idx) for idx in encoder_layer_range]
b_enc_list = ["b_enc_" + str(idx) for idx in encoder_layer_range]

#Std weights for last encoder layer
w_enc_std = ["w_enc_" + str(num_encoder_layers-1) + "_std"]
b_enc_std = ["b_enc_" + str(num_encoder_layers-1) + "_std"]

w_dec_list = ["w_dec_" + str(idx) for idx in encoder_layer_range[::-1]]
b_dec_list = ["b_dec_" + str(idx) for idx in encoder_layer_range[::-1]]

weights_list = w_enc_list + w_dec_list
bias_list = b_enc_list + b_dec_list

#Train list is ordered by input to output weights, then input to output bias
train_list = weights_list + bias_list

num_train_weights = len(train_list)

schedule = [
  {"weights": train_list,
  "decay_mult": 0.0,
  "sparse_mult": 0.0,
  "kld_mult": 1/params["batch_size"],
  "weight_lr": [0.0001,]*num_train_weights,
  "decay_steps": [int(3e6*0.4),]*num_train_weights,
  "decay_rate": [0.5,]*num_train_weights,
  "staircase": [True,]*num_train_weights,
  "num_batches": int(3e5)}]
