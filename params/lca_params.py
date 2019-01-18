import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of LCA neurons
      num_steps    [int] Number of inference steps
      dt           [float] Discrete global time constant
      tau          [float] LCA time constant
      thresh_type  [str] "hard" or "soft" - LCA threshold function specification
    """
    super(params, self).__init__()
    self.model_type = "lca"
    self.model_name = "lca"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False # FT whitening already does LPF
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.batch_size = 100
    self.num_neurons = 768
    self.num_steps = 50
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "annealed_sgd"
    self.cp_int = int(1e4)
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_ver = "0.0"
    self.cp_load_step = None # latest checkpoint
    #self.cp_load_var = ["phi"]
    self.log_int = int(1e2)
    self.log_to_file = True
    self.gen_plot_int = int(1e4)
    self.save_plots = True
    self.schedule = [
      {"weights": None,#["weights/w:0"],
      "num_batches": int(1e4),
      "sparse_mult": 0.1,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.5),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.num_neurons = 768
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.25
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["num_batches"] = int(2e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.7*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.5

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 150
      self.vectorize_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.whiten_method = "FT"
      self.extract_patches = True
      self.num_neurons = 768
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.8
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["num_batches"] = int(2e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    elif data_type.lower() == "field":
      self.model_name += "_field"
      self.vectorize_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.extract_patches = True
      self.num_neurons = 768
      self.thresh_type = "soft"
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.8
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["num_batches"] = int(2e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

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
