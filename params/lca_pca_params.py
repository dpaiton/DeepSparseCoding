import os
from params.lca_params import params as lca_params

class params(lca_params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_pooling_units [int] number of 2nd layer units
    """
    super(params, self).__init__()
    self.model_type = "lca_pca"
    self.model_name = "lca_pca"
    self.version = "0.0"
    self.batch_size = 100
    self.num_neurons = 768
    self.num_pooling_units = 192
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_step = int(1e6) # None gives latest checkpoint
    self.cp_load_name = "lca_pca_512_vh_ft_white"
    self.cp_load_ver = "0.0"
    #self.cp_load_var = ["phi"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "sparse_mult": 1.0,
      "weight_lr": [0.01],
      "num_batches": 1e5,
      "decay_steps": [int(1e5*0.5)],
      "decay_rate": [0.8],
      "staircase": [True]}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = False
      self.standardize_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.rectify_a = True
      self.num_neurons = 1536
      self.thresh_type = "soft"
      self.cp_int = int(1e5)
      self.gen_plot_int = int(1e5)
      self.log_int = int(1e2)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(5e5)
        self.schedule[sched_idx]["sparse_mult"] = 0.3#0.15
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["decay_steps"] = int(0.7*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.5

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.num_neurons = 768
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.21
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
    self.num_steps = 5
