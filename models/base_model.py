import os
import numpy as np
import tensorflow as tf
from utils.logger import Logger
import utils.data_processing as dp
from utils.trainable_variable_dict import TrainableVariableDict

class Model(object):
  def __init__(self):
    self.optimizers_added = False
    self.savers_constructed = False
    self.params_loaded = False
    self.trainable_variables = TrainableVariableDict()
    self.graph = tf.Graph()
    self.full_model_load_ignore = []

  def setup(self, params):
    self.load_params(params)
    #self.check_params()
    self.make_dirs()
    self.init_logging()
    self.log_params()
    self.add_step_counter_to_graph()
    self.build_graph()
    self.log_trainable_variables()
    self.load_schedule(params.schedule)
    self.sched_idx = 0
    self.log_schedule()
    self.add_optimizers_to_graph()
    self.add_initializer_to_graph()
    self.construct_savers()

  def reset_graph(self):
    self.graph = tf.Graph()
    self.trainable_variables = TrainableVariableDict()
    self.full_model_load_ignore = []

  def check_schedule_type(self, val, target_type, target_len):
    if (type(val) == list):
      assert len(val) == target_len
      out_val = val
    else: #scalar is used
      out_val = [val,] * target_len
    #Check type
    for v in out_val:
      assert type(v) == target_type
    return out_val

  def load_schedule(self, schedule):
    """
    Load schedule into object
    Inputs:
     schedule: [list of dict] learning schedule
    """
    for sched in schedule:
      assert type(sched["num_batches"]) == int
      if sched["weights"] is not None: # schedule specificies specific variables for trainable vars
        assert type(sched["weights"]) == list
      else: # scalar is used
        #assert type(sched["weight_lr"]) == float
        #assert type(sched["decay_steps"]) == int
        #assert type(sched["decay_rate"]) == float
        #assert type(sched["staircase"]) == bool
        sched["weights"] = self.get_trainable_variable_names()

      target_len = len(sched["weights"])
      sched["weight_lr"] = self.check_schedule_type(sched["weight_lr"], float, target_len)
      sched["decay_steps"] = self.check_schedule_type(sched["decay_steps"], int, target_len)
      sched["decay_rate"] = self.check_schedule_type(sched["decay_rate"], float, target_len)
      sched["staircase"] = self.check_schedule_type(sched["staircase"], bool, target_len)

  def get_trainable_variable_names(self):
    return list(self.trainable_variables.keys())

  def load_params(self, params):
    """
    Calculates a few extra parameters
    Sets parameters as member variable
    """
    params.cp_latest_filename = "latest_checkpoint_v"+params.version
    params.cp_load_latest_filename = "latest_checkpoint_v"+params.cp_load_ver
    params.cp_load_dir = params.out_dir + params.cp_load_name+ "/checkpoints/"
    if not hasattr(params, "model_out_dir"):
      params.model_out_dir = params.out_dir + params.model_name
    params.cp_save_dir = params.model_out_dir + "/checkpoints/"
    params.log_dir = params.model_out_dir + "/logfiles/"
    params.save_dir = params.model_out_dir + "/savefiles/"
    params.disp_dir = params.model_out_dir + "/vis/"
    params.num_pixels = int(np.prod(params.data_shape))
    self.params = params
    self.params_loaded = True

  def check_params(self):
    """Check parameters with assertions"""
    raise NotImplementedError

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

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.params.model_out_dir):
      os.makedirs(self.params.model_out_dir)
    if not os.path.exists(self.params.log_dir):
      os.makedirs(self.params.log_dir)
    if not os.path.exists(self.params.cp_save_dir):
      os.makedirs(self.params.cp_save_dir)
    if not os.path.exists(self.params.save_dir):
      os.makedirs(self.params.save_dir)
    if not os.path.exists(self.params.disp_dir):
      os.makedirs(self.params.disp_dir)

  def init_logging(self, log_filename=None):
    if self.params.log_to_file:
      if log_filename is None:
        log_filename = self.params.log_dir+self.params.model_name+"_v"+self.params.version+".log"
        self.logger = Logger(filename=log_filename, overwrite=True)
    else:
        self.logger = Logger(filename=None)

  def js_dumpstring(self, obj):
    """Dump json string with special NumpyEncoder"""
    return self.logger.js_dumpstring(obj)

  def log_params(self, params=None):
    """Use logging to write model params"""
    if params is not None:
      dump_obj = params.__dict__
    else:
      dump_obj = self.params.__dict__
    self.logger.log_params(dump_obj)

  def log_trainable_variables(self):
    """Use logging to write model params"""
    var_names = list(self.trainable_variables.keys())
    self.logger.log_trainable_variables(var_names)

  def log_schedule(self):
    """Use logging to write current schedule, as specified by self.sched_idx"""
    self.logger.log_schedule(self.params.schedule)

  def log_info(self, string):
    """Log input string"""
    self.logger.log_info(string)

  def compute_weight_gradients(self, optimizer, weight_op=None):
    """Returns the gradients for a weight variable using a given optimizer"""
    if self.params.optimizer == "lbfgsb":
      return [(optimizer, weight_op)]
    return optimizer.compute_gradients(self.get_total_loss(), var_list=weight_op)

  def add_optimizers_to_graph(self):
    """
    Add optimizers to graph
    Creates member variables grads_and_vars and apply_grads for each weight
      - both member variables are indexed by [schedule_idx][weight_idx]
      - grads_and_vars holds the gradients for the weight updates
      - apply_grads is the operator to be called to perform weight updates
    TODO: For BFGS - Could use the step_callback and loss_callback args to have functions that get
      grads (loss_callback) and also increment the global step (step_callback)
      Also, for BFGS the minimizer requires a session as input to minimize, not sure the best way
      to minimize in the sch_grads_and_vars list?
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("optimizers") as scope:
          self.grads_and_vars = list() # [sch_idx][weight_idx]
          self.apply_grads = list() # [sch_idx][weight_idx]
          self.learning_rates = list() # [sch_idx][weight_idx]
          if self.params.optimizer == "lbfgsb":
            self.minimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss,
              options={"maxiter":self.params.maxiter}) # Default method is L-BFGSB
          for schedule_idx, sch in enumerate(self.params.schedule):
            sch_grads_and_vars = list() # [weight_idx]
            sch_apply_grads = list() # [weight_idx]
            sch_lrs = list() # [weight_idx]
            #Construct weight ops
            weight_ops = [self.trainable_variables[weight] for weight in sch["weights"]]
            for w_idx, weight in enumerate(sch["weights"]):
              weight_name = weight.split("/")[-1].split(":")[0]
              learning_rates = tf.train.exponential_decay(
                learning_rate=sch["weight_lr"][w_idx],
                global_step=self.global_step,
                decay_steps=sch["decay_steps"][w_idx],
                decay_rate=sch["decay_rate"][w_idx],
                staircase=sch["staircase"][w_idx],
                name="annealing_schedule_"+weight_name)
              sch_lrs.append(learning_rates)
              if self.params.optimizer == "annealed_sgd": # TODO: rename to "sgd"
                optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                  name="grad_optimizer_"+weight_name)
              elif self.params.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rates, beta1=0.9, beta2=0.99,
                  epsilon=1e-07, name="adam_optimizer_"+weight_name)
              elif self.params.optimizer == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rates, epsilon=1e-07,
                  name="adadelta_optimizer_"+weight_name)
              elif self.params.optimizer == "lbfgsb":
                optimizer = None
              else:
                assert False, ("Optimizer "+self.params.optimizer+" is not supported.")
              weight_op = self.trainable_variables[weight]
              sch_grads_and_vars.append(self.compute_weight_gradients(optimizer, weight_op))
              gstep = self.global_step if w_idx == 0 else None # Only increment once
              if self.params.optimizer == "lbfgsb": # BFGS doesn't actually need the update op
                if w_idx == 0:
                  sch_apply_grads.append(tf.assign_add(self.global_step, 1))
                else:
                  sch_apply_grads.append(None)
              else:
                sch_apply_grads.append(optimizer.apply_gradients(sch_grads_and_vars[w_idx],
                  global_step=gstep))
            self.learning_rates.append(sch_lrs)
            self.grads_and_vars.append(sch_grads_and_vars)
            self.apply_grads.append(sch_apply_grads)
    self.optimizers_added = True

  def add_initializer_to_graph(self):
    """
    Add initializer to the graph
    This must be done after optimizers have been added
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("initialization") as scope:
          self.init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())

  def get_load_vars(self):
    """Get variables for loading"""
    all_vars = tf.global_variables()
    if self.params.cp_load_var is None:
      load_v = [v for v in all_vars if v not in self.full_model_load_ignore]
    else:
      load_v = []
      for weight in self.params.cp_load_var:
        found=False
        for var in all_vars:
          if var.name == weight:
            load_v.append(var)
            found=True
            break
        if not found:
          assert False, ("Weight specified in cp_load_var "+str(weight)+" not found")
    return load_v

  def construct_savers(self):
    """Add savers to graph"""
    with self.graph.as_default():
      #Loop through to find variables to ignore
      all_vars = tf.global_variables()
      load_vars = [v for v in all_vars if v not in self.full_model_load_ignore]
      self.full_saver = tf.train.Saver(max_to_keep=self.params.max_cp_to_keep,
        var_list=load_vars)
      if self.params.cp_load:
        self.loader = tf.train.Saver(var_list=self.get_load_vars())
    self.savers_constructed = True

  def write_saver_defs(self):
    """Write saver definitions for full model and weights-only"""
    assert self.savers_constructed
    full_saver_def = self.full_saver.as_saver_def()
    full_file = self.params.save_dir+self.params.model_name+"_v"+self.params.version+".def"
    with open(full_file, "wb") as f:
      f.write(full_saver_def.SerializeToString())
    self.logger.log_info("Full saver def saved in file %s"%full_file)

  def write_graph(self, graph_def):
    """Write graph structure to protobuf file"""
    write_name = self.params.model_name+"_v"+self.params.version+".pb"
    self.writer = tf.summary.FileWriter(self.params.save_dir, graph=self.graph)
    tf.train.write_graph(graph_def,
      logdir=self.params.save_dir, name=write_name, as_text=False)
    self.logger.log_info("Graph def saved in file %s"%self.params.save_dir+write_name)

  def write_checkpoint(self, session):
    """Write checkpoints for full model and weights-only"""
    base_save_path = self.params.cp_save_dir+self.params.model_name+"_v"+self.params.version
    full_save_path = self.full_saver.save(session,
      save_path=base_save_path,
      global_step=self.global_step,
      latest_filename=self.params.cp_latest_filename)
    self.logger.log_info("Full model saved in file %s"%full_save_path)
    return base_save_path

  def load_checkpoint(self, session, model_dir):
    """
    Load checkpoint model into session.
    Inputs:
      session: tf.Session() that you want to load into
      model_dir: String specifying the path to the checkpoint
    """
    assert self.params.cp_load == True, ("cp_load must be set to true to load a checkpoint")
    self.loader.restore(session, model_dir)

  def load_full_model(self, session, model_dir):
    """
    Load checkpoint model into session.
    Inputs:
      session: tf.Session() that you want to load into
      model_dir: String specifying the path to the checkpoint
    """
    self.full_saver.restore(session, model_dir)

  def get_schedule(self, key=None):
    """
    Returns the current schedule being executed
    Inputs:
      key: [str] key in dictionary
    """
    if key is not None:
      assert key in self.params.schedule[self.sched_idx].keys(), (
        key+" was not found in the schedule.")
      return self.params.schedule[self.sched_idx][key]
    return self.params.schedule[self.sched_idx]

  def set_schedule(self, key, val):
    """
    Modifies the internal schedule for the current schedule index
    Inputs:
      key: [str] key in dictionary
      val: value be set in schedlue,
        if there is already a val for key, new val must be of same type
    """
    if key in self.params.schedule[self.sched_idx].keys():
      assert type(val) == type(self.params.schedule[self.sched_idx][key]), (
        "val must have type "+str(type(self.params.schedule[self.sched_idx][key])))
    self.params.schedule[self.sched_idx][key] = val

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    """
    Return dictionary containing all placeholders
    Inputs:
      input_data: data to be placed in self.input_placeholder
      input_labels: label to be placed in self.label_placeholder
      dict_args: optional dictionary to be appended to the automatically generated feed_dict
    """
    placeholders = [op.name
      for op
      in self.graph.get_operations()
      if ("auto_placeholders" in op.name
      and "input_data" not in op.name
      and "input_label" not in op.name)
      ]
    if input_labels is not None and hasattr(self, "label_placeholder"):
      feed_dict = {self.input_placeholder:input_data, self.label_placeholder:input_labels}
    else:
      feed_dict = {self.input_placeholder:input_data}
    for placeholder in placeholders:
      feed_dict[self.graph.get_tensor_by_name(placeholder+":0")] = (
        self.get_schedule(placeholder.split("/")[1]))
    if dict_args is not None:
      feed_dict.update(dict_args)
    return feed_dict

  # Functions that expose specific variables to outer classes
  # Subclass must overwrite this class
  # TODO: is it possible to compute these here in the base class?
  def get_input_shape(self):
    return NotImplementedError

  def get_num_latent(self):
    return NotImplementedError

  def get_total_loss(self):
    raise NotImplementedError

  def get_encodings(self):
    raise NotImplementedError

  def build_input_placeholder(self):
    with tf.device(self.params.device):
      with self.graph.as_default():
        self.input_placeholder = tf.placeholder(tf.float32,
          shape=self.get_input_shape(), name="input_data")
    return self.input_placeholder

  def normalize_input(self, input_node):
    with tf.device(self.params.device):
      with self.graph.as_default():
        #Normalize here if using tf_standardize_data
        if(self.params.tf_standardize_data):
          out_img = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
            input_node)
        else:
          out_img = input_node
    return out_img

  def add_step_counter_to_graph(self):
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

  #If build_graph gets called without parameters,
  #will build placeholder first, then call build_graph_from_input
  #Subclasses can overwrite this function to ignore this functionality
  def build_graph(self):
    input_node = self.build_input_placeholder()
    input_node = self.normalize_input(input_node)
    self.build_graph_from_input(input_node)

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    raise NotImplementedError

  def slice_features(self, input, indices):
    """
    Slice input array along last dimension using indices
      gather_nd only gets first indices,
      so we permute the last index (or "features', which we want to slice into) to the first index
    Inputs:
      input [Tensor] of any shape
      indices [list or scalar] indices for slicing
    Outputs:
      sliced_input [Tensor] where the last index is sliced using indices
    """
    t_input = tf.transpose(input)
    gather_idxs = np.array([[i] for i in indices]).astype(np.int32)
    t_actual = tf.gather_nd(t_input, gather_idxs)
    actual = tf.transpose(t_actual)
    return actual

  def reshape_dataset(self, dataset, params):
    """
    Reshape dataset to fit model expectations
    Inputs:
      dataset [dict] returned from data/data_picker
    """
    assert hasattr(params, "vectorize_data"), (
      "Model params must set vectorize_data.")
    for key in dataset.keys():
      if dataset[key] is None:
        continue
      dataset[key].images = dp.reshape_data(dataset[key].images, params.vectorize_data)[0]
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
      lpf_data: low pass filter data with a Gaussian kernel
      center_data: subtract mean from data
      norm_data: divide data by the maximum
      rescale_data: rescale data to be between 0 and 1
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
      if dataset[key] is None:
        continue
      if hasattr(params, "whiten_data") and params.whiten_data:
        if hasattr(params, "whiten_method"):
          if params.whiten_method == "FT": # other methods require patching first
            dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
              dp.whiten_data(dataset[key].images, method=params.whiten_method)
            print("Preprocessing: FT Whitened "+key+" data")
      if hasattr(params, "lpf_data") and params.lpf_data:
        dataset[key].images, dataset[key].data_mean, dataset[key].lpf_filter = \
          dp.lpf_data(dataset[key].images, cutoff=params.lpf_cutoff)
        print("Preprocessing: Low pass filtered "+key+" data")
      if hasattr(params, "contrast_normalize") and params.contrast_normalize:
        if hasattr(params, "gauss_patch_size"):
          dataset[key].images = dp.contrast_normalize(dataset[key].images,
            params.gauss_patch_size)
        else:
          dataset[key].images = dp.contrast_normalize(dataset[key].images)
        print("Preprocessing: Contrast normalized "+key+" data")
      if hasattr(params, "standardize_data") and params.standardize_data:
        dataset[key].images, dataset[key].data_mean, dataset[key].data_std = \
          dp.standardize_data(dataset[key].images)
        self.data_mean = dataset[key].data_mean
        self.data_std = dataset[key].data_std
        print("Preprocessing: Standardized "+key+" data")
      if hasattr(params, "extract_patches") and params.extract_patches:
        assert all(key in params.__dict__.keys()
          for key in ["num_patches", "patch_edge_size", "overlapping_patches",
          "randomize_patches"]), ("Insufficient params for patches.")
        out_shape = (int(params.num_patches), int(params.patch_edge_size),
          int(params.patch_edge_size), dataset[key].num_channels)
        dataset[key].num_examples = out_shape[0]
        dataset[key].reset_counters()
        if hasattr(params, "patch_variance_threshold"):
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params.overlapping_patches, params.randomize_patches,
            params.patch_variance_threshold, dataset[key].rand_state)
        else:
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params.overlapping_patches, params.randomize_patches,
            var_thresh=0, rand_state=dataset[key].rand_state)
        dataset[key].shape = dataset[key].images.shape
        dataset[key].num_rows = dataset[key].shape[1]
        dataset[key].num_cols = dataset[key].shape[2]
        dataset[key].num_channels = dataset[key].shape[3]
        dataset[key].num_pixels = np.prod(dataset[key].shape[1:])
        print("Preprocessing: Extracted patches from "+key+" data")
        if hasattr(params, "whiten_data") and params.whiten_data:
          if hasattr(params, "whiten_method") and params.whiten_method != "FT":
            dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
              dp.whiten_data(dataset[key].images, method=params.whiten_method)
            print("Preprocessing: Whitened "+key+" data")
      if hasattr(params, "norm_data") and params.norm_data:
        dataset[key].images, dataset[key].data_max = dp.normalize_data_with_max(dataset[key].images)
        self.data_max = dataset[key].data_max
        print("Preprocessing: Normalized "+key+" data with maximum")
      if hasattr(params, "rescale_data") and params.rescale_data:
        dataset[key].images, dataset[key].data_min, dataset[key].data_max = dp.rescale_data_to_one(dataset[key].images)
        self.data_max = dataset[key].data_max
        self.data_min = dataset[key].data_min
        print("Preprocessing: Normalized "+key+" data with maximum")
      if hasattr(params, "center_data") and params.center_data:
        dataset[key].images, dataset[key].data_mean = dp.center_data(dataset[key].images,
          use_dataset_mean=True)
        self.data_mean = dataset[key].data_mean
        print("Preprocessing: Centered "+key+" data")
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
    update_dict = self.generate_update_dict(input_data, input_labels, batch_step)
    js_str = self.js_dumpstring(update_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
      """
      Generates a dictionary to be logged in the print_update function
      """
      update_dict = dict()
      return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, gradients, etc
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    pass

  #Labels are needed by adv examples
  def evaluate_model_batch(self, batch_size, images, labels=None,
    var_names=None, var_nodes=None):
    """
    Creates a session with the loaded model graph to run all tensors specified by var_names
    Runs in batches
    Outputs:
      evals [dict] containing keys that match var_names or var_node (depending on which
        gets specified) and the values computed from the session run
      Note that all var_names/var_nodes must have batch dimension in first dimension
    Inputs:
      batch_size scalar that defines the batch size to split images up into
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      var_names [list of str] list of strings containing the tf variable names to be evaluated
      var_nodes [list of tf nodes] list of tensorflow node references containing the
        variables to be evaluated
    """
    num_data = images.shape[0]
    num_iterations = int(np.ceil(num_data / batch_size))

    evals = {}

    assert images.shape[0] == labels.shape[0], (
      "Images and labels must be the same shape, not %g and %g"%(images.shape[0], labels.shape[0]))
    #^ is xor
    assert (var_names is None) ^ (var_nodes is None),  \
      ("Only one of var_names or var_nodes can be specified")

    if(var_names is not None):
      dict_keys = var_names
      tensors = [self.graph.get_tensor_by_name(name) for name in var_names]
    elif(var_nodes is not None):
      dict_keys = var_nodes
      tensors = var_nodes

    for key in dict_keys:
      evals[key] = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.get_default_session()
    for it in range(num_iterations):
      batch_start_idx = int(it * batch_size)
      batch_end_idx = int(np.min([batch_start_idx + batch_size, num_data]))
      batch_images = images[batch_start_idx:batch_end_idx, ...]
      if(labels is not None):
        batch_labels = labels[batch_start_idx:batch_end_idx, ...]
      else:
        batch_labels = None

      feed_dict = self.get_feed_dict(batch_images, input_labels=batch_labels, is_test=True)
      sch = self.get_schedule()
      #TODO: (see train_model.py)
      #if("train_on_adversarial" in sch):
      #  if(sch["train_on_adversarial"]):
      #    self.modify_input(feed_dict)
      if("train_on_adversarial" not in sch):
        sch["train_on_adversarial"] = False
      self.modify_input(feed_dict, sch["train_on_adversarial"])
      eval_list = sess.run(tensors, feed_dict)

      for key, ev in zip(dict_keys, eval_list):
        evals[key].append(ev)

    #Concatenate all evals in batch dim
    for key, val in evals.items():
      evals[key] = np.concatenate(val, axis=0)
    return evals

  def modify_input(self, feed_dict, train_on_adversarial):
    return feed_dict
