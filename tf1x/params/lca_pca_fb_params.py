from DeepSparseCoding.tf1x.params.lca_pca_params import params as lca_pca_params

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
      {"weights": None,
      "num_batches": int(1e4),
      "sparse_mult": 0.08,
      "fb_mult": 1e-3,
      "weight_lr": [0.8],
      "decay_steps": [int(1e4*0.6)],
      "decay_rate": [0.5],
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
        self.schedule[sched_idx]["fb_mult"] = 1e-3
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["decay_steps"] = int(0.7*self.schedule[sched_idx]["num_batches"])
        self.schedule[sched_idx]["decay_rate"] = 0.5

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.vectorize_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.whiten_method = "FT"
      self.whiten_batch_size = 10
      self.extract_patches = True
      self.num_neurons = 768
      self.num_steps = 50
      self.thresh_type = "soft"
      self.cp_int = int(1e5)
      self.log_int = int(1e2)
      self.gen_plot_int = int(2e4)
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.3
        self.schedule[sched_idx]["fb_mult"] = 1e-3
        self.schedule[sched_idx]["weight_lr"] = 0.01
        self.schedule[sched_idx]["num_batches"] = int(1e5)
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
        self.schedule[sched_idx]["sparse_mult"] = 0.3
        self.schedule[sched_idx]["fb_mult"] = 1e-3
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
        self.schedule[sched_idx]["fb_mult"] = 1e-3
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
