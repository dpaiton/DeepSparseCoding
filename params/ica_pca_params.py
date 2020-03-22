from DeepSparseCoding.params.ica_params import params as ica_params

class params(ica_params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_pooling_units [int] number of 2nd layer units
    """
    super(params, self).__init__()
    self.model_type = "ica_pca"
    self.model_name = "ica_pca"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.whiten_data = True
    self.contrast_normalize = False
    self.extract_patches = True
    self.num_patches = 1e5
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 1e-6
    self.batch_size = 100
    self.prior = "laplacian" # "cauchy"
    self.optimizer = "sgd"
    self.num_pooling_units = 50
    self.cp_int = 10000
    self.max_cp_to_keep = 2
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = 150000
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["a"]
    self.log_int = 500
    self.log_to_file = True
    self.gen_plot_int = 500
    self.save_plots = True
    self.schedule = [
      {"weights": None,
      "weight_lr": [0.01],
      "num_batches": int(1e4),
      "decay_steps": [int(1e4*0.8)],
      "decay_rate": [0.7],
      "staircase": [True]}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if self.data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.batch_size = 50
      self.vectorize_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.whiten_method = "ZCA"
      self.lpf_data = True # only for ZCA/PCA
      self.lpf_cutoff = 0.7
      self.whiten_batch_size = 10
      self.extract_patches = True
      self.patch_edge_size = 16
      self.thresh_type = "soft"
      self.cp_int = int(1e5)
      self.log_int = int(1e2)
      self.gen_plot_int = int(2e4)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(5e5)
        self.schedule[sched_idx]["weight_lr"] = 0.001
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    elif self.data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_patches = 50
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
