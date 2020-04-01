from DeepSparseCoding.params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      maxiter [int] maximum number of iterations of lbfgs optimizer
    """
    super(params, self).__init__()
    self.model_type = "rica"
    self.model_name = "rica_768"
    self.version = "0.0"
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = False
    self.rescale_data = False
    self.standardize_data = False
    self.tf_standardize_data = False
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
    self.batch_size = 100
    self.num_neurons = 768
    self.optimizer = "lbfgsb" #"adam"#"sgd"
    self.maxiter = 15000
    self.cp_int = 100000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "rica_pretrain" # trained with sgd
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = None
    self.log_int = 10
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True
    self.schedule = [
      {"num_batches": 1,
      "weights": None,
      "recon_mult": 0.8,
      "sparse_mult": 1.0,
      "weight_lr": 0.3,
      "decay_steps": 1,
      "decay_rate": 0.5,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.version = "0.0"
      self.batch_size = 50000
      self.vectorize_data = True
      self.rescale_data = True
      self.standardize_data = False
      self.whiten_data = False
      self.whiten_method = "ZCA"
      self.lpf_data = True
      self.lpf_cutoff = 0.7
      self.extract_patches = False
      self.num_neurons = 768
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["recon_mult"] = 0.8
        self.schedule[sched_idx]["sparse_mult"] = 1.0
        self.schedule[sched_idx]["weight_lr"] = 0.3
        self.schedule[sched_idx]["num_batches"] = 1
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.batch_size = int(1e5)
      self.vectorize_data = True
      self.center_data = True
      self.rescale_data = False
      self.whiten_data = True
      self.whiten_method = "ZCA"
      self.lpf_data = True
      self.lpf_cutoff = 0.7
      self.extract_patches = True
      self.num_neurons = 768
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["recon_mult"] = 0.8
        self.schedule[sched_idx]["sparse_mult"] = 1.0
        self.schedule[sched_idx]["weight_lr"] = 0.3
        self.schedule[sched_idx]["num_batches"] = 1
        self.schedule[sched_idx]["decay_steps"] = int(0.9*self.schedule[sched_idx]["num_batches"])
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
        self.schedule[sched_idx]["recon_mult"] = 0.8
        self.schedule[sched_idx]["sparse_mult"] = 1.0
        self.schedule[sched_idx]["weight_lr"] = 0.3
        self.schedule[sched_idx]["num_batches"] = 1
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["num_batches"] = 1
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
