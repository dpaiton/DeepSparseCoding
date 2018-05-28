import os
import numpy as np
import tensorflow as tf
from utils.logger import Logger
import utils.data_processing as dp

class Model(object):
  def __init__(self):
    self.optimizers_added = False
    self.savers_constructed = False
    self.vector_inputs = None
    self.params_loaded = False

  def setup(self, params, schedule):
    self.load_schedule(schedule)
    self.sched_idx = 0
    self.load_params(params)
    self.check_params()
    self.make_dirs()
    self.init_logging()
    self.log_params()
    self.log_schedule()
    self.setup_graph()

  def load_schedule(self, schedule):
    """
    Load schedule into object
    Inputs:
     schedule: [list of dict] learning schedule
    """
    for sched in schedule:
      assert len(sched["weights"]) == len(sched["weight_lr"])
      assert len(sched["weights"]) == len(sched["decay_steps"])
      assert len(sched["weights"]) == len(sched["decay_rate"])
      assert len(sched["weights"]) == len(sched["staircase"])
    self.sched = schedule

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      model_type     [str] Type of model
                       Can be "MLP", "karklin_lewicki"
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
    self.params = params
    # Meta-parameters
    self.model_name = str(params["model_name"])
    self.model_type = str(params["model_type"])
    if "num_labeled" in params.keys():
      self.num_labeled = str(params["num_labeled"])
    if "num_unlabeled" in params.keys():
      self.num_unlabeled = str(params["num_unlabeled"])
    self.version = str(params["version"])
    self.optimizer = str(params["optimizer"])
    # Output generation
    self.log_int = int(params["log_int"])
    self.log_to_file = bool(params["log_to_file"])
    self.gen_plot_int = int(params["gen_plot_int"])
    self.save_plots = bool(params["save_plots"])
    # Checkpointing
    self.cp_int = int(params["cp_int"])
    self.max_cp_to_keep = int(params["max_cp_to_keep"])
    self.cp_load = bool(params["cp_load"])
    if self.cp_load:
      self.cp_load_name = str(params["cp_load_name"])
      if "cp_load_step" in params and params["cp_load_step"] is not None:
        self.cp_load_step = int(params["cp_load_step"])
      else:
        self.cp_load_step = None
      self.cp_load_ver = str(params["cp_load_ver"])
      if "cp_load_var" in params:
        self.cp_load_var = [str(var) for var in params["cp_load_var"]]
      else:
        self.cp_load_var = []
      if "cp_set_var" in params:
        self.cp_set_var = [str(var) for var in params["cp_set_var"]]
      else:
        self.cp_set_var = []
      self.cp_load_dir = (str(params["out_dir"]) + self.cp_load_name
        + "/checkpoints/")
    # Directories
    self.out_dir = str(params["out_dir"])
    if "model_out_dir" in params.keys():
      self.model_out_dir = params["model_out_dir"]
    else:
      self.model_out_dir = self.out_dir + self.model_name
    self.cp_save_dir = self.model_out_dir + "/checkpoints/"
    self.log_dir = self.model_out_dir + "/logfiles/"
    self.save_dir = self.model_out_dir + "/savefiles/"
    self.disp_dir = self.model_out_dir + "/vis/"
    # Other
    self.eps = float(params["eps"])
    self.device = str(params["device"])
    self.rand_seed = int(params["rand_seed"])
    self.params_loaded = True

  def check_params(self):
    """Check parameters with assertions"""
    pass

  def get_param(self, param_name):
    """
    Get param value from model
      This is equivalent to self.param_name, except that it will return None if
      the param does not exist.
    """
    if hasattr(self, param_name):
      return getattr(self, param_name)
    else:
      return None

  def set_param(self, param_name, new_value):
    """
    Modifies a model parameter
    Inputs:
      param_name: [str] parameter name, must already exist
      new_value: [] new parameter value, must be the same type as old param value
    """
    assert hasattr(self, param_name)
    assert type(getattr(self, param_name)) == type(new_value)
    setattr(self, param_name, new_value)

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.model_out_dir):
      os.makedirs(self.model_out_dir)
    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
    if not os.path.exists(self.cp_save_dir):
      os.makedirs(self.cp_save_dir)
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    if not os.path.exists(self.disp_dir):
      os.makedirs(self.disp_dir)

  def init_logging(self, log_filename=None):
    if self.log_to_file:
      if log_filename is None:
        log_filename = self.log_dir+self.model_name+"_v"+self.version+".log"
        self.logger = Logger(log_filename, overwrite=True)
    else:
        self.logger = Logger()

  def js_dumpstring(self, obj):
    """Dump json string with special NumpyEncoder"""
    return self.logger.js_dumpstring(obj)

  def log_params(self, params=None):
    """Use logging to write model params"""
    if params is not None:
      dump_obj = params
    else:
      dump_obj = self.params
    self.logger.log_params(dump_obj)

  def log_schedule(self):
    """Use logging to write current schedule, as specified by self.sched_idx"""
    self.logger.log_schedule(self.sched)

  def log_info(self, string):
    """Log input string"""
    self.logger.log_info(string)

  def compute_weight_gradients(self, optimizer, weight_op=None):
    """Returns the gradients for a weight variable using a given optimizer"""
    return optimizer.compute_gradients(self.total_loss, var_list=weight_op)

  def add_optimizers_to_graph(self):
    """
    Add optimizers to graph
    Creates member variables grads_and_vars and apply_grads for each weight
      - both member variables are indexed by [schedule_idx][weight_idx]
      - grads_and_vars holds the gradients for the weight updates
      - apply_grads is the operator to be called to perform weight updates
    """
    with self.graph.as_default():
      with tf.name_scope("optimizers") as scope:
        self.grads_and_vars = list() # [sch_idx][weight_idx]
        self.apply_grads = list() # [sch_idx][weight_idx]
        for schedule_idx, sch in enumerate(self.sched):
          sch_grads_and_vars = list() # [weight_idx]
          sch_apply_grads = list() # [weight_idx]
          for w_idx, weight in enumerate(sch["weights"]):
            learning_rates = tf.train.exponential_decay(
              learning_rate=sch["weight_lr"][w_idx],
              global_step=self.global_step,
              decay_steps=sch["decay_steps"][w_idx],
              decay_rate=sch["decay_rate"][w_idx],
              staircase=sch["staircase"][w_idx],
              name="annealing_schedule_"+weight)
            if self.optimizer == "annealed_sgd":
              optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                name="grad_optimizer_"+weight)
            elif self.optimizer == "adam":
              optimizer = tf.train.AdamOptimizer(learning_rates, beta1=0.9, beta2=0.99,
                epsilon=1e-07, name="adam_optimizer_"+weight)
            elif self.optimizer == "adadelta":
              optimizer = tf.train.AdadeltaOptimizer(learning_rates, epsilon=1e-07,
                name="adadelta_optimizer_"+weight)
            with tf.variable_scope("weights", reuse=True) as scope:
              weight_op = [tf.get_variable(weight)]
            sch_grads_and_vars.append(self.compute_weight_gradients(optimizer, weight_op))
            gstep = self.global_step if w_idx == 0 else None # Only increment once
            sch_apply_grads.append(optimizer.apply_gradients(sch_grads_and_vars[w_idx],
              global_step=gstep))
          self.grads_and_vars.append(sch_grads_and_vars)
          self.apply_grads.append(sch_apply_grads)
    self.optimizers_added = True

  def add_initializer_to_graph(self):
    """
    Add initializer to the graph
    This must be done after optimizers have been added
    """
    assert self.optimizers_added
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("initialization") as scope:
          self.init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())

  def get_load_vars(self):
    """Get variables for loading"""
    all_vars = tf.global_variables()
    if len(self.cp_load_var) > 0:
      load_v = [var
        for var in all_vars
        for weight in self.cp_load_var
        if weight in var.name]
      assert len(load_v) > 0, ("Weights specified by cp_load_var not found.")
    else:
      load_v = all_vars
    return load_v

  def construct_savers(self):
    """Add savers to graph"""
    assert self.optimizers_added, (
      "Optimizers must be added to the graph before constructing savers.")
    with self.graph.as_default():
      with tf.variable_scope("weights", reuse=True) as scope:
        weights = [weight for weight in tf.global_variables()
          if weight.name.startswith(scope.name)]
      self.weight_saver = tf.train.Saver(var_list=weights,
        max_to_keep=self.max_cp_to_keep)
      self.full_saver = tf.train.Saver(max_to_keep=self.max_cp_to_keep)
      if self.cp_load:
        self.loader = tf.train.Saver(var_list=self.get_load_vars())
    self.savers_constructed = True

  def write_saver_defs(self):
    """Write saver definitions for full model and weights-only"""
    assert self.savers_constructed
    full_saver_def = self.full_saver.as_saver_def()
    full_file = self.save_dir+self.model_name+"_v"+self.version+"-full.def"
    with open(full_file, "wb") as f:
      f.write(full_saver_def.SerializeToString())
    self.logger.log_info("Full saver def saved in file %s"%full_file)
    weight_saver_def = self.weight_saver.as_saver_def()
    weight_file = self.save_dir+self.model_name+"_v"+self.version+"-weights.def"
    with open(weight_file, "wb") as f:
      f.write(weight_saver_def.SerializeToString())
    self.logger.log_info("Weight saver def saved in file %s"%weight_file)

  def write_graph(self, graph_def):
    """Write graph structure to protobuf file"""
    write_name = self.model_name+"_v"+self.version+".pb"
    self.writer = tf.summary.FileWriter(self.save_dir, graph=self.graph)
    tf.train.write_graph(graph_def,
      logdir=self.save_dir, name=write_name, as_text=False)
    self.logger.log_info("Graph def saved in file %s"%self.save_dir+write_name)

  def write_checkpoint(self, session):
    """Write checkpoints for full model and weights-only"""
    base_save_path = self.cp_save_dir+self.model_name+"_v"+self.version
    full_save_path = self.full_saver.save(session,
      save_path=base_save_path+"_full",
      global_step=self.global_step,
      latest_filename="latest_checkpoint_v"+self.version)
    self.logger.log_info("Full model saved in file %s"%full_save_path)
    weight_save_path = self.weight_saver.save(session,
      save_path=base_save_path+"_weights",
      global_step=self.global_step,
      latest_filename="latest_checkpoint_v"+self.version)
    self.logger.log_info("Weights model saved in file %s"%weight_save_path)
    return base_save_path

  def load_full_model(self, session, model_dir):
    """
    Load checkpoint model into session.
    Inputs:
      session: tf.Session() that you want to load into
      model_dir: String specifying the path to the checkpoint
    """
    self.full_saver.restore(session, model_dir)

  def load_weights(self, session, model_dir):
    """
    Load checkpoint weights into session.
    Inputs:
      session: tf.Session() that you want to load into
      model_dir: String specifying the path to the checkpoint
    """
    self.weight_saver.restore(session, model_dir)

  def get_schedule(self, key=None):
    """
    Returns the current schedule being executed
    Inputs:
      key: [str] key in dictionary
    """
    if key is not None:
      assert key in self.sched[self.sched_idx].keys(), (
        key+" was not found in the schedule.")
      return self.sched[self.sched_idx][key]
    return self.sched[self.sched_idx]

  def set_sched(self, key, val):
    """
    Modifies the internal schedule for the current schedule index
    Inputs:
      key: [str] key in dictionary
      val: value be set in schedlue,
        if there is already a val for key, new val must be of same type
    """
    if key in self.sched[self.sched_idx].keys():
      assert type(val) == type(self.sched[self.sched_idx][key]), (
        "val must have type "+str(type(self.sched[self.sched_idx][key])))
    self.sched[self.sched_idx][key] = val

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None):
    """
    Return dictionary containing all placeholders
    Inputs:
      input_data: data to be placed in self.x
      input_labels: label to be placed in self.y
      dict_args: optional dictionary to be appended to the automatically generated feed_dict
    """
    placeholders = [op.name
      for op
      in self.graph.get_operations()
      if ("placeholders" in op.name
      and "input_data" not in op.name
      and "input_label" not in op.name)]
    if input_labels is not None and hasattr(self, "y"):
      feed_dict = {self.x:input_data, self.y:input_labels}
    else:
      feed_dict = {self.x:input_data}
    for placeholder in placeholders:
      feed_dict[self.graph.get_tensor_by_name(placeholder+":0")] = (
        self.get_schedule(placeholder.split("/")[1]))
    if dict_args is not None:
      feed_dict.update(dict_args)
    return feed_dict

  def setup_graph(self):
    """Setup graph object and add optimizers, initializer"""
    self.graph = tf.Graph()
    self.build_graph()
    self.add_optimizers_to_graph()
    self.add_initializer_to_graph()
    self.construct_savers()

  def build_graph(self):
    """Build the TensorFlow graph object"""
    pass

  def reshape_dataset(self, dataset, params):
    """
    Reshape dataset to fit model expectations
    Inputs:
      dataset [dict] returned from data/data_picker
    """
    assert self.vector_inputs is not None, (
      "Model object must set vector_inputs member variable in __init__() method.")
    for key in dataset.keys():
      if "data_shape" in params.keys():
        data_shape = params["data_shape"]
      else:
        data_shape = None
      dataset[key].images = dp.reshape_data(dataset[key].images, self.vector_inputs, data_shape)[0]
      dataset[key].shape = dataset[key].images.shape
    return dataset

  def preprocess_dataset(self, dataset, params=None):
    """
    This is a wrapper function to demonstrate how preprocessing can be performed.
    Inputs:
      dataset [dict] returned from data/data_picker
      params [dict] kwargs for preprocessing, if None then the member variable is used
    Parameters are set using the model parameter dictionary.
    Possible parameters  are:
      center_data: subtract mean from data
      norm_data: divide data by the maximum
      whiten_data: default method is using the Fourier amplitude spectrium ("FT")
        change default with whiten_method param
      standardize_data: subtract mean and divide by the standard deviation
      contrast_normalize: divide by gaussian blurred surround pixels
      extract_patches: break up data into patches
        see utils/data_processing/exract_patches() for docs
    """
    if params is None:
      assert self.params_loaded, (
        "You must either provide parameters or load the model params before preprocessing.")
      params = self.params
    for key in dataset.keys():
      if "center_data" in params.keys() and params["center_data"]:
        dataset[key].images, dataset[key].data_mean = dp.center_data(dataset[key].images,
          use_dataset_mean=True)
        self.data_mean = dataset[key].data_mean
      if "whiten_data" in params.keys() and params["whiten_data"]:
        if "whiten_method" in params.keys():
          if params["whiten_method"] == "FT": # other methods require patching first
            dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
              dp.whiten_data(dataset[key].images, method=params["whiten_method"])
      if "lpf_data" in params.keys() and params["lpf_data"]:
        dataset[key].images, dataset[key].data_mean, dataset[key].lpf_filter = \
          dp.lpf_data(dataset[key].images, cutoff=params["lpf_cutoff"])
      if "contrast_normalize" in params.keys() and params["contrast_normalize"]:
        if "gauss_patch_size" in params.keys():
          dataset[key].images = dp.contrast_normalize(dataset[key].images,
            params["gauss_patch_size"])
        else:
          dataset[key].images = dp.contrast_normalize(dataset[key].images)
      if "standardize_data" in params.keys() and params["standardize_data"]:
        dataset[key].images, dataset[key].data_mean, dataset[key].data_std = \
          dp.standardize_data(dataset[key].images)
        self.data_mean = dataset[key].data_mean
        self.data_std = dataset[key].data_std
      if "extract_patches" in params.keys() and params["extract_patches"]:
        assert all(key in params.keys()
          for key in ["num_patches", "patch_edge_size", "overlapping_patches",
          "randomize_patches"]), ("Insufficient params for patches.")
        out_shape = (int(params["num_patches"]), int(params["patch_edge_size"]),
          int(params["patch_edge_size"]), dataset[key].num_channels)
        dataset[key].num_examples = out_shape[0]
        dataset[key].reset_counters()
        if "patch_variance_threshold" in params.keys():
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params["overlapping_patches"], params["randomize_patches"],
            params["patch_variance_threshold"], dataset[key].rand_state)
        else:
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params["overlapping_patches"], params["randomize_patches"],
            var_thresh=0, rand_state=dataset[key].rand_state)
        dataset[key].shape = dataset[key].images.shape
        dataset[key].num_rows = dataset[key].shape[1]
        dataset[key].num_cols = dataset[key].shape[2]
        dataset[key].num_channels = dataset[key].shape[3]
        dataset[key].num_pixels = np.prod(dataset[key].shape[1:])
        if "whiten_data" in params.keys() and params["whiten_data"]:
          if "whiten_method" in params.keys() and params["whiten_method"] != "FT":
            dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
              dp.whiten_data(dataset[key].images, method=params["whiten_method"])
      if "norm_data" in params.keys() and params["norm_data"]:
        dataset[key].images, dataset[key].data_max = dp.normalize_data_with_max(dataset[key].images)
    return dataset

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    NOTE: For the analysis code to parse update statistics, the self.js_dumpstring() call
      must receive a dict object. Additionally, the self.js_dumpstring() output must be
      logged with <stats> </stats> tags.
      For example: logging.info("<stats>"+self.js_dumpstring(output_dictionary)+"</stats>")
    """
    pass

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, gradients, etc
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    pass
