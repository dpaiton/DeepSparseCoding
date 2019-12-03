import os
from params.lca_pca_params import params as lca_pca_params

class params(lca_pca_params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_pooling_units [int] indicating the number of 2nd layer units
      act_cov_loc [str] file path indicating the location of the activity covariance
        computed from analysis.
    """
    super(params, self).__init__()
    self.model_type = "lca_pca_fb"
    self.model_name = "lca_pca_fb"
    self.version = "0.0"
    self.batch_size = 100
    self.num_neurons = 512
    self.num_pooling_units = 50
    self.num_steps = 50
    self.rectify_a = True
    self.thresh_type = "soft"
    self.optimizer = "sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_step = int(1e6) # None gives latest checkpoint
    self.cp_load_name = "pretrain"
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["phi"]
    self.log_int = 1
    self.log_to_file = True
    self.gen_plot_int = 100
    self.save_plots = True
    self.schedule = [
      {"weights": ["lca/weights/w:0"],
      "num_batches": int(1e4),
      "sparse_mult": 0.08,
      "fb_mult": 1e-3,
      "weight_lr": [0.8],
      "decay_steps": [int(1e4*0.6)],
      "decay_rate": [0.5],
      "staircase": [True]}]
