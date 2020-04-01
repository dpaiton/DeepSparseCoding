from DeepSparseCoding.tf1x.params.ica_params import params as ica_params

class params(ica_params):
  def __init__(self):
    super(params, self).__init__()
    # model config
    self.model_type = "ica_subspace"
    self.model_name = "ica_subspace"
    self.version = "0.0"
    self.vectorize_data = True
    self.whiten_data = True
    self.whiten_method = "ZCA"
    self.lpf_data = True # only for ZCA/PCA
    self.lpf_cutoff = 0.625
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.batch_size = 50
    self.prior = "laplacian" #"cauchy"
    self.optimizer = "sgd"
    self.cp_int = 1000
    self.log_int = 300
    self.log_to_file = True
    self.gen_plot_int = 1000
    self.save_plots = True
    self.schedule = [
      {"weights": ["weights/w_analy:0"],
      "num_batches": int(3e5),
      "weight_lr": 0.001,
      "decay_steps": int(5e5*0.8),
      "decay_rate": 0.8,
      "staircase": True}]


  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 100
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = False
      self.center_data = True
      self.whiten_data = True
      self.whiten_method = "ZCA"
      self.extract_patches = True
      self.whiten_batch_size = 10
      self.num_patches = int(1e5)
      self.batch_size = 1000
      self.patch_edge_size = 16
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      self.orthonorm_weights = True
      self.num_neurons = 256
      self.num_pixels = 256
      self.num_groups = 64
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.2

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_neurons = 256
      self.num_pixels = 256
      self.num_groups = 64

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_patches = 50
    self.num_neurons = 256
    self.num_pixels = 256
    self.num_groups = 64
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4

