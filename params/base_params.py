import os
import numpy as np

class BaseParams(object):
  """
  Modifiable parameters:
    model_type     [str] Type of model
    model_name     [str] Name for model
    out_dir        [str] Base directory where output will be directed
    version        [str] Model version for output
                           default is an empty string ""
    optimizer      [str] Which optimization algorithm to use
                     Can be "annealed_sgd" (default) or "adam"
    log_int        [int] How often to send updates to stdout
    log_to_file    [bool] If set, log to file, else log to stderr
    gen_plot_int   [int] How often to generate plots
    save_plots     [bool] If set, save plots to file
    cp_int         [int] How often to checkpoint
    max_cp_to_keep [int] How many checkpoints to keep. See max_to_keep tf arg
    cp_load        [bool] if set, load from checkpoint
    cp_load_name   [str] Checkpoint model name to load
    cp_load_step   [int] Checkpoint time step to load
    cp_load_ver    [str] Checkpoint version to load
    cp_load_var    [list of str] which variables to load
                     if None or empty list, the model load all weights
    cp_set_var     [list of str] which variables to assign values to
                     len(cp_set_var) should equal len(cp_load_var)
    eps            [float] Small value to avoid division by zero
    device         [str] Which device to run on
    rand_seed      [int] Random seed
  """
  def __init__(self):
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
    self.batch_size = 100
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.val_on_cp = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.eps = 1e-12
    self.device = "/gpu:0"
    self.rand_seed = 123456789
    self.rand_state = np.random.RandomState(self.rand_seed)
    self.out_dir = os.path.expanduser("~")+"/Work/Projects/"
    self.data_dir = os.path.expanduser("~")+"/Work/Datasets/"

  #def set_data_params(self, data_type):
  #  pass

  #def set_test_params(self, data_type):
  #  pass

