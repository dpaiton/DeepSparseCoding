import os
from params.base_params import Base_Params
from models.vae import VAE as vae

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      TODO
    """
    super(params, self).__init__()
    self.model_type = "vae"
    self.model_name = "vae"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = True
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.num_batches = int(3e5)
    self.batch_size = 200
    #Specify number of neurons for encoder
    #Last element in list is the size of the latent space
    #Decoder will automatically build the transpose of the encoder
    self.output_channels = [512, 50]
    self.optimizer = "adam"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
    else:
      assert False, ("Data type "+data_type+" is not supported.")

trainable_vars = vae().get_trainable_variable_names(params())
num_train_weights = len(trainable_vars)

schedule = [
  {"weights": trainable_vars,
  "decay_mult": 0.0,
  "sparse_mult": 0.0,
  "kld_mult": 1/params().batch_size,
  "weight_lr": [0.0001,]*num_train_weights,
  "decay_steps": [int(3e6*0.4),]*num_train_weights,
  "decay_rate": [0.5,]*num_train_weights,
  "staircase": [True,]*num_train_weights}]

