import os
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      maxiter [int] maximum number of iterations of lbfgs optimizer
    """
    super(params, self).__init__()
    self.model_type = "rica"
    self.model_name = "rica"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = True
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "ZCA"
    self.lpf_data = True
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.num_batches = 1
    self.batch_size = 100000
    self.num_neurons = 768
    self.norm_weights = False
    self.optimizer = "lbfgsb" #"adam"#"annealed_sgd"
    self.maxiter = 15000
    self.cp_int = 100000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "rica_pretrain" # trained with sgd
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w"]
    self.log_int = 10
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True
    self.schedule = [
      {"weights": ["w"],
      "recon_mult": 0.8,
      "sparse_mult": 1.0,
      "weight_lr": [0.3],
      "decay_steps": [int(self.num_batches*0.9)],
      "decay_rate": [0.5],
      "staircase": [True]}]
