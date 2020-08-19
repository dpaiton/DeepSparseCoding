import os

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

from DeepSparseCoding.tf1x.utils.logger import Logger
from DeepSparseCoding.tf1x.utils import plot_functions as pf
from DeepSparseCoding.tf1x.models import model_picker as mp
from DeepSparseCoding.tf1x.utils import data_processing as dp
from DeepSparseCoding.tf1x.data.dataset import Dataset
from DeepSparseCoding.tf1x.modules.class_adversarial_module import ClassAdversarialModule
from DeepSparseCoding.tf1x.modules.recon_adversarial_module import ReconAdversarialModule
from DeepSparseCoding.tf1x.modules.neuron_visualization_module import NeuronVisualizationModule

class Analyzer(object):
  def setup(self, input_params):
    """
    Clobbering:
      if user wants to clobber:
        remove old log file if it exists
        create new log file object
        log analysis params
        set params member variable to input params
      else:
        if the file exists and is not empty
          load previous text
          extract previous params
            (if there are multiple entries, extract the last one)
          merge new params into previous params (new overwrites previous)
          create log file object with append set (overwrite=False)
          set params member variable to merged params
        else:
          create new log file object
          set params member variable to input params
        log merged analysis params
    """
    # Load model parameters and schedule
    self.model_log_file = (input_params.model_dir+"/logfiles/"+input_params.model_name
      +"_v"+input_params.version+".log")
    self.model_logger = Logger(self.model_log_file, overwrite=False)
    self.model_log_text = self.model_logger.load_file()
    self.model_params = self.model_logger.read_params(self.model_log_text)[-1]
    self.model_schedule = self.model_logger.read_schedule(self.model_log_text)
    # Load or create analysis params log
    self.analysis_out_dir = input_params.model_dir+"/analysis/"+input_params.version+"/"
    self.make_dirs() # If analysis log does not exist then we want to make the folder first
    self.analysis_log_file = (self.analysis_out_dir+"/logfiles/"+input_params.save_info+".log")
    if input_params.overwrite_analysis_log:
      if os.path.exists(self.analysis_log_file):
        os.remove(self.analysis_log_file)
      self.analysis_logger = Logger(self.analysis_log_file, overwrite=True)
      self.analysis_params = input_params
    else:
      if os.path.exists(self.analysis_log_file) and os.stat(self.analysis_log_file).st_size != 0:
        self.analysis_logger = Logger(self.analysis_log_file, overwrite=False)
        analysis_text = self.analysis_logger.load_file()
        prev_analysis_params = self.analysis_logger.read_params(analysis_text)[-1]
        for attr_key in input_params.__dict__.keys(): # overwrite the previous params with new params
          setattr(prev_analysis_params, attr_key, getattr(input_params, attr_key))
        self.analysis_params = prev_analysis_params
      else:
        self.analysis_logger = Logger(self.analysis_log_file, overwrite=True)
        self.analysis_params = input_params
    self.analysis_params.cp_loc = tf.train.latest_checkpoint(self.model_params.cp_save_dir,
      latest_filename="latest_checkpoint_v"+self.analysis_params.version)
    self.model_params.model_out_dir = self.analysis_out_dir # prevent model from clobbering training
    self.check_params()
    self.rand_state = np.random.RandomState(self.analysis_params.rand_seed)
    self.analysis_logger.log_params(self.analysis_params.__dict__)
    self.get_model() # Adds "self.model" member variable that is another model class

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.analysis_out_dir):
      os.makedirs(self.analysis_out_dir)
    if not os.path.exists(self.analysis_out_dir+"/savefiles"):
      os.makedirs(self.analysis_out_dir+"/savefiles")
    if not os.path.exists(self.analysis_out_dir+"/logfiles"):
      os.makedirs(self.analysis_out_dir+"/logfiles")

  def check_params(self):
    if not hasattr(self.analysis_params, "device"):
      self.analysis_params.device = self.model_params.device
    if hasattr(self.analysis_params, "data_dir"):
      self.model_params.data_dir = self.analysis_params.data_dir
    if not hasattr(self.analysis_params, "rand_seed"):
      self.analysis_params.rand_seed = self.model_params.rand_seed
    if not hasattr(self.analysis_params, "input_scale"):
      self.analysis_params.input_scale = 1.0
    # Evaluate model variables on images
    if not hasattr(self.analysis_params, "do_evals"):
      self.analysis_params.do_evals = False
    # Training run analysis
    if not hasattr(self.analysis_params, "do_run_analysis"):
      self.analysis_params.do_run_analysis = False
    # BF fits
    if not hasattr(self.analysis_params, "do_basis_analysis"):
      self.analysis_params.do_basis_analysis = False
    if not hasattr(self.analysis_params, "ft_padding"):
      self.analysis_params.ft_padding = None
    if not hasattr(self.analysis_params, "num_gauss_fits"):
      self.analysis_params.num_gauss_fits = 20
    if not hasattr(self.analysis_params, "gauss_thresh"):
      self.analysis_params.gauss_thresh = 0.2
    # Activity Triggered Averages
    if not hasattr(self.analysis_params, "do_atas"):
      self.analysis_params.do_atas = False
    if not hasattr(self.analysis_params, "num_ata_images"):
      self.analysis_params.num_ata_images = 100
    if not hasattr(self.analysis_params, "num_noise_images"):
      self.analysis_params.num_noise_images = 100
    # Recon Adversarial analysis
    if hasattr(self.analysis_params, "do_recon_adversaries"):
      if not hasattr(self.analysis_params, "adversarial_step_size"):
        self.analysis_params.adversarial_step_size = 0.01
      if not hasattr(self.analysis_params, "adversarial_num_steps"):
        self.analysis_params.adversarial_num_steps = 200
      if not hasattr(self.analysis_params, "adversarial_input_id"):
        self.analysis_params.adversarial_input_id = None
      if not hasattr(self.analysis_params, "adversarial_target_id"):
        self.analysis_params.adversarial_target_id = None
    else:
      self.analysis_params.do_recon_adversaries = False
    # Class Adversarial analysis
    if hasattr(self.analysis_params, "do_class_adversaries"):
      if not hasattr(self.analysis_params, "adversarial_step_size"):
        self.analysis_params.adversarial_step_size = 0.01
      if not hasattr(self.analysis_params, "adversarial_num_steps"):
        self.analysis_params.adversarial_num_steps = 200
      if not hasattr(self.analysis_params, "adversarial_input_id"):
        self.analysis_params.adversarial_input_id = None
    else:
      self.analysis_params.do_class_adversaries = False

  def get_model(self):
    """Load model object into analysis object"""
    self.model = mp.get_model(self.model_params.model_type)

  # If build_graph gets called without parameters,
  # build placeholders call build graph with default input
  def build_graph(self):
    # TODO: adversaries & neuron vis should all be able to be True.
    # Need to build in better abstraction for specifying these analyses
    # We want to overwrite model adversarial params with what we have in analysis
    if(self.analysis_params.do_class_adversaries):
      assert self.analysis_params.do_recon_adversaries is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      assert self.analysis_params.do_neuron_visualization is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          self.input_node = self.model.build_input_placeholder()
          with tf.compat.v1.variable_scope("placeholders") as scope:
            # This is a switch used internally to use clean or adv examples
            self.use_adv_input = tf.compat.v1.placeholder(tf.bool, shape=(), name="use_adv_input")
          # Building adv module here with adv_params
          self.class_adv_module = ClassAdversarialModule(
            self.input_node,
            self.use_adv_input,
            self.model_params.num_classes,
            self.analysis_params.adversarial_num_steps,
            self.analysis_params.adversarial_step_size,
            max_step=self.analysis_params.adversarial_max_change,
            clip_adv=self.analysis_params.adversarial_clip,
            clip_range=self.analysis_params.adversarial_clip_range,
            attack_method=self.analysis_params.adversarial_attack_method,
            eps=self.model_params.eps)
      self.input_node = self.model.normalize_input(self.class_adv_module.adv_image)
      self.model.build_graph_from_input(self.input_node)
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          self.class_adv_module.build_adversarial_ops(
            label_est=self.model.get_label_est(),
            model_logits=self.model.get_logits_with_temp(),
            label_gt=self.model.label_placeholder)
      #Add adv module ignore list to model ignore list
      self.model.full_model_load_ignore.extend(self.class_adv_module.ignore_load_var_list)
    elif(self.analysis_params.do_recon_adversaries):
      assert self.analysis_params.do_class_adversaries is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      assert self.analysis_params.do_neuron_visualization is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          self.input_node = self.model.build_input_placeholder()
          with tf.compat.v1.variable_scope("placeholders") as scope:
            # This is a switch used internally to use clean or adv examples
            self.use_adv_input = tf.compat.v1.placeholder(tf.bool, shape=(), name="use_adv_input")
          self.recon_adv_module = ReconAdversarialModule(
            data_tensor=self.input_node,
            use_adv_input=self.use_adv_input,
            num_steps=self.analysis_params.adversarial_num_steps,
            step_size=self.analysis_params.adversarial_step_size,
            max_adv_change=self.analysis_params.adversarial_max_change,
            clip_adv=self.analysis_params.adversarial_clip,
            clip_range=self.analysis_params.adversarial_clip_range,
            attack_method=self.analysis_params.adversarial_attack_method,
            carlini_change_variable=self.analysis_params.carlini_change_variable,
            adv_optimizer=self.analysis_params.adv_optimizer)
      # Add adv module ops to the graph by passing the module outputs to the model graph as input
      self.input_node = self.model.normalize_input(self.recon_adv_module.adv_images)
      self.model.build_graph_from_input(self.input_node)
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          self.recon_adv_module.build_adversarial_ops(
            recons=self.model.reconstruction,
            latent_activities=self.model.get_encodings())
      # Add adv module ignore list to model ignore list
      self.model.full_model_load_ignore.extend(self.recon_adv_module.ignore_load_var_list)
    elif(self.analysis_params.do_neuron_visualization):
      assert self.analysis_params.do_class_adversaries is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      assert self.analysis_params.do_recon_adversaries is False, (
        "Only one of do_class_adversaries, do_recon_adversaries, do_neuron_visualization"
        +" can be True.")
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          self.input_node = self.model.build_input_placeholder()
          self.neuron_vis_module = NeuronVisualizationModule(
            data_tensor=self.input_node,
            num_steps=self.analysis_params.neuron_vis_num_steps, # int
            step_size=self.analysis_params.neuron_vis_step_size, # int
            clip_output=self.analysis_params.neuron_vis_clip, # bool
            clip_range=self.analysis_params.neuron_vis_clip_range, # [float, float]
            norm_constraint_mag=self.analysis_params.neuron_vis_norm_magnitude, # None or float
            l2_regularize_coeff=self.analysis_params.neuron_vis_l2_regularize_coeff, # None or float
            variation_coeff=self.analysis_params.neuron_vis_variation_coeff, # None or float
            method=self.analysis_params.neuron_vis_method, # str
            optimizer=self.analysis_params.neuron_vis_optimizer) # str
      # Add vis module ops to the graph by passing the module outputs to the model graph as input
      self.input_node = self.model.normalize_input(self.neuron_vis_module.vis_image)
      self.model.build_graph_from_input(self.input_node)
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          if self.analysis_params.neuron_vis_target_layer is None:
            target_layer = self.model.get_encodings()
          else:
            target_layer = self.model.module.u_list[self.analysis_params.neuron_vis_target_layer]
          self.neuron_vis_module.build_visualization_ops(target_layer)
      # Add vis module ignore list to model ignore list
      self.model.full_model_load_ignore.extend(self.neuron_vis_module.ignore_load_var_list)
    else:
      self.model.build_graph()

  def set_input_transform_func(self, func):
    self.input_transform_func = func

  def setup_model(self, params):
    """
    Run model setup, but also add adversarial nodes to graph
    """
    self.model.load_params(params)
    #self.model.check_params()
    self.model.make_dirs()
    self.model.init_logging()
    self.model.log_params()
    self.model.add_step_counter_to_graph()
    #Call own build graph to call model's build graph
    self.build_graph()
    self.model.load_schedule(params.schedule)
    self.model.sched_idx = 0
    self.model.log_schedule()
    self.add_pre_init_ops_to_graph()
    self.model.add_optimizers_to_graph()
    self.model.add_initializer_to_graph()
    self.model.construct_savers()

  def add_pre_init_ops_to_graph(self):
    pass

  def run_analysis(self, images, labels=None, save_info=""):
    """
    Wrapper function for running all available model analyses
    Log statistics should be consistent across models, but in general it is expected that
    this method will be overwritten for specific models
    """
    if self.analysis_params.do_run_analysis:
      self.run_stats = self.stats_analysis(save_info)

  def load_analysis(self, save_info=""):
    # Run statistics
    stats_file_loc = self.analysis_out_dir+"savefiles/run_stats_"+save_info+".npz"
    if os.path.exists(stats_file_loc):
      self.run_stats = np.load(stats_file_loc, allow_pickle=True)["data"].item()["run_stats"]
    # var_names evaluated
    eval_file_loc = self.analysis_out_dir+"savefiles/evals_"+save_info+".npz"
    if os.path.exists(eval_file_loc):
      self.evals = np.load(eval_file_loc, allow_pickle=True)["data"].item()["evals"]
    # Basis function fits
    try:
      self.load_basis_stats(save_info)
    except FileNotFoundError:
      self.analysis_logger.log_info("WARNING: Basis stats file not found")
    # Activity triggered analysis
    ata_file_loc = self.analysis_out_dir+"savefiles/atas_"+save_info+".npz"
    if os.path.exists(ata_file_loc):
      ata_analysis = np.load(ata_file_loc, allow_pickle=True)["data"].item()
      self.atas = ata_analysis["atas"]
      self.atcs = ata_analysis["atcs"]
    ata_noise_file_loc = self.analysis_out_dir+"savefiles/atas_noise_"+save_info+".npz"
    if os.path.exists(ata_noise_file_loc):
      ata_noise_analysis = np.load(ata_noise_file_loc, allow_pickle=True)["data"].item()
      self.noise_atas = ata_noise_analysis["noise_atas"]
      self.noise_atcs = ata_noise_analysis["noise_atcs"]
    act_noise_file_loc = self.analysis_out_dir+"savefiles/noise_response_"+save_info+".npz"
    if os.path.exists(act_noise_file_loc):
      noise_analysis = np.load(act_noise_file_loc, allow_pickle=True)["data"].item()
      self.noise_activity = noise_analysis["noise_activity"]
      self.analysis_params.num_noise_images = self.noise_activity.shape[0]
    # Orientation analysis
    tuning_file_locs = [self.analysis_out_dir+"savefiles/ot_responses_"+save_info+".npz",
      self.analysis_out_dir+"savefiles/co_responses_"+save_info+".npz"]
    if os.path.exists(tuning_file_locs[0]):
      self.ot_grating_responses = np.load(tuning_file_locs[0], allow_pickle=True)["data"].item()
    if os.path.exists(tuning_file_locs[1]):
      self.co_grating_responses = np.load(tuning_file_locs[1], allow_pickle=True)["data"].item()
    recon_file_loc = self.analysis_out_dir+"savefiles/full_recon_"+save_info+".npz"
    if os.path.exists(recon_file_loc):
      recon_analysis = np.load(recon_file_loc, allow_pickle=True)["data"].item()
      self.full_image = recon_analysis["full_image"]
      self.full_recon = recon_analysis["full_recon"]
      self.recon_frac_act = recon_analysis["recon_frac_act"]

    #TODO: Smarter naming scheme for save_info (e.g. how it is done for models)
    # Recon Adversarial analysis
    recon_adversarial_stats_file_loc = (
      self.analysis_out_dir+"savefiles/recon_adversary_stats_"+save_info+".npz")
    if os.path.exists(recon_adversarial_stats_file_loc):
      data = np.load(recon_adversarial_stats_file_loc, allow_pickle=True)["data"].item()
      self.steps_idx = data["steps_idx"]
      self.recon_adversarial_input_images = data["input_images"]
      self.adversarial_target_images = data["target_images"]
      self.adversarial_images = data["adversarial_images"]
      self.analysis_params.adversarial_step_size = data["step_size"]
      self.analysis_params.adversarial_num_steps = data["num_steps"]
      self.num_data = data["num_data"]
      self.analysis_params.adversarial_input_id = data["input_id"]
      self.analysis_params.adversarial_target_id = data["target_id"]
      self.adversarial_input_target_mses = data["input_target_mse"]
      self.adversarial_input_recon_mses = data["input_recon_mses"]
      self.adversarial_input_adv_mses = data["input_adv_mses"]
      self.adversarial_target_recon_mses = data["target_recon_mses"]
      self.adversarial_target_adv_mses = data["target_adv_mses"]
      self.adversarial_adv_recon_mses = data["adv_recon_mses"]
      self.adversarial_target_adv_cos_similarities = data["target_adv_cos_similarities"]
      self.adversarial_input_adv_cos_similarities = data["input_adv_cos_similarities"]
      self.adversarial_target_pert_cos_similarities = data["target_pert_cos_similarities"]
      self.adversarial_input_pert_cos_similarities = data["input_pert_cos_similarities"]
    recon_adversarial_file_loc = (
      self.analysis_out_dir+"savefiles/recon_adversary_recons_"+save_info+".npz")
    if os.path.exists(recon_adversarial_file_loc):
      data = np.load(recon_adversarial_file_loc, allow_pickle=True)["data"].item()
      self.adversarial_recons = data["adversarial_recons"]

    #Class adversarial analysis
    class_adversarial_file_loc = (
      self.analysis_out_dir+"savefiles/class_adversary_"+save_info+".npz")
    if os.path.exists(class_adversarial_file_loc):
      data = np.load(class_adversarial_file_loc, allow_pickle=True)["data"].item()
      self.steps_idx = data["steps_idx"]
      self.class_adversarial_input_images = data["input_images"]
      self.adversarial_input_labels = data["input_labels"]
      self.adversarial_target_labels = data["target_labels"]
      self.adversarial_outputs = data["adversarial_outputs"]
      self.analysis_params.adversarial_step_size = data["step_size"]
      self.analysis_params.adversarial_num_steps = data["num_steps"]
      self.num_data = data["num_data"]
      self.analysis_params.adversarial_input_id = data["input_id"]
      self.adversarial_input_adv_mses = data["input_adv_mses"]
      self.adversarial_target_output_losses = data["target_output_losses"]
      self.adversarial_clean_accuracy = data["clean_accuracy"]
      self.adversarial_adv_accuracy = data["adv_accuracy"]
      self.adversarial_success_rate = data["attack_success_rate"]
    class_adversarial_file_loc = \
      self.analysis_out_dir+"savefiles/class_adversary_images_"+save_info+".npz"
    if os.path.exists(class_adversarial_file_loc):
      data = np.load(class_adversarial_file_loc, allow_pickle=True)["data"].item()
      self.adversarial_images = data["adversarial_images"]

    # Optimal stimulus analysis
    neuron_visualization_file_loc = (
      self.analysis_out_dir+"savefiles/neuron_visualization_analysis_"+save_info+".npz")
    if os.path.exists(neuron_visualization_file_loc):
      self.neuron_vis_output = np.load(neuron_visualization_file_loc, allow_pickle=True)["data"].item()

  def load_basis_stats(self, save_info):
    bf_file_loc = self.analysis_out_dir+"savefiles/basis_"+save_info+".npz"
    self.bf_stats = np.load(bf_file_loc, allow_pickle=True)["data"].item()["bf_stats"]

  def stats_analysis(self, save_info):
    """Run stats extracted from the logfile"""
    run_stats = self.get_log_stats()
    np.savez(self.analysis_out_dir+"savefiles/run_stats_"+save_info+".npz",
      data={"run_stats":run_stats})
    self.analysis_logger.log_info("Run stats analysis is complete.")
    return run_stats

  def get_log_stats(self):
    """Wrapper function for parsing the log statistics"""
    return self.model_logger.read_stats(self.model_log_text)

  def eval_analysis(self, images, var_names, save_info):
    evals = self.evaluate_model(images, var_names)
    np.savez(self.analysis_out_dir+"savefiles/evals_"+save_info+".npz", data={"evals":evals})
    self.analysis_logger.log_info("Image analysis is complete.")
    return evals

  def evaluate_model_batch(self, batch_size, images, var_names):
    #TODO have this function call model's evaluate_model_batch
    """
    Creates a session with the loaded model graph to run all tensors specified by var_names
    Runs in batches
    Outputs:
      evals [dict] containing keys that match var_names and the values computed from the session run
      Note that all var_names must have batch dimension in first dimension
    Inputs:
      batch_size scalar that defines the batch size to split images up into
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      var_names [list of str] list of strings containing the tf variable names to be evaluated
    """
    num_data = images.shape[0]
    num_iterations = int(np.ceil(num_data / batch_size))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    evals = {}
    for name in var_names:
      evals[name] = []
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      image_shape = (batch_size,) + images.shape[1:]
      sess.run(self.model.init_op, {self.model.input_placeholder:np.zeros(image_shape)})
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      for it in range(num_iterations):
        batch_start_idx = int(it * batch_size)
        # TODO: I think this will break if batch_size does not divide evenly into num_data
        #   because the init op has set a batch_size
        batch_end_idx = int(np.min([batch_start_idx + batch_size, num_data]))
        batch_images = images[batch_start_idx:batch_end_idx, ...]
        feed_dict = self.model.get_feed_dict(batch_images, is_test=True)
        eval_list = sess.run(tensors, feed_dict)
        for name, ev in zip(var_names, eval_list):
          evals[name].append(ev)
    #Concatenate all evals in batch dim
    for key, val in evals.items():
      evals[key] = np.concatenate(val, axis=0)
    return evals

  def evaluate_model(self, images, var_names):
    """
    Creates a session with the loaded model graph to run all tensors specified by var_names
    Outputs:
      evals [dict] containing keys that match var_names and the values computed from the session run
    Inputs:
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      var_names [list of str] list of strings containing the tf variable names to be evaluated
    TODO: Rewrite to not take in images, use for evaluating non-batch variables (e.g. weights)
    evaluate_model_batch will take in images and evaluate batch variables
    """
    feed_dict = self.model.get_feed_dict(images, is_test=True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def evaluate_tf_tensor(self, tensor, feed_dict):
    """
    Creates a session with the loaded model graph to run variable
    Outputs:
      tensor_eval [np.ndarray] containing the values computed from the session run
    Inputs:
      tensor [tf variable] variable to be evaluated
      feed_dict [dict] feed dictionary with required keys for the input tensor
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      tensor_eval = sess.run(tensor, feed_dict)
    return tensor_eval

  def basis_analysis(self, weights, save_info):
    bf_stats = dp.get_dictionary_stats(weights, padding=self.analysis_params.ft_padding,
      num_gauss_fits=self.analysis_params.num_gauss_fits,
      gauss_thresh=self.analysis_params.gauss_thresh)
    np.savez(self.analysis_out_dir+"savefiles/basis_"+save_info+".npz", data={"bf_stats":bf_stats})
    self.analysis_logger.log_info("Dictionary analysis is complete.")
    return bf_stats

  def ata_analysis(self, images, activity, save_info):
    atas = self.compute_atas(activity, images)
    atcs = self.compute_atcs(activity, images, atas)
    np.savez(self.analysis_out_dir+"savefiles/atas_"+save_info+".npz",
      data={"atas":atas, "atcs":atcs})
    self.analysis_logger.log_info("Activity triggered analysis is complete.")
    return (atas, atcs)

  def run_noise_analysis(self, save_info, batch_size=100):
    """
    Computes activations and  activity triggered averages & covariances on Gaussian noise images
    """
    noise_activity = []
    noise_image_list = []
    num_images_processed = 0
    while num_images_processed < self.analysis_params.num_noise_images:
      if batch_size + num_images_processed <= self.analysis_params.num_noise_images:
        noise_shape = [batch_size] + self.model_params.data_shape
        num_images_processed += batch_size
      else:
        noise_shape = [self.analysis_params.num_noise_images - num_images_processed] \
          + self.model_params.data_shape
        num_images_processed = self.analysis_params.num_noise_images
      noise_images = self.rand_state.standard_normal(noise_shape)
      noise_image_list.append(noise_images)
      noise_activity.append(self.compute_activations(noise_images))
    noise_images = np.concatenate(noise_image_list, axis=0)
    noise_activity = np.concatenate(noise_activity, axis=0)
    noise_atas, noise_atcs = self.ata_analysis(noise_images, noise_activity, "noise_"+save_info)
    np.savez(self.analysis_out_dir+"savefiles/noise_responses_"+save_info+".npz",
      data={"num_noise_images":self.analysis_params.num_noise_images,
      "noise_activity":noise_activity})
    self.analysis_logger.log_info("Noise analysis is complete.")
    return (noise_activity, noise_atas, noise_atcs)

  def compute_activations(self, images, batch_size=None, activation_operation=None):
    """
    Computes the output code for a set of images.
    Outputs:
      evaluated activation_operation on the input images
    Inputs:
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      batch_size [int] how many inputs to use in a batch
      activation_operation [tf operation] that produces the output activation
        if None then it defaults to `self.model.get_encodings()`
    """
    if activation_operation is None:
        activation_operation = self.model.get_encodings
    images_shape = list(images.shape)
    num_images = images_shape[0]
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      if batch_size is not None and batch_size < num_images:
        assert num_images % batch_size == 0, (
          "batch_size=%g must divide evenly into num_images=%g"%(batch_size, num_images))
        num_batches = int(np.ceil(num_images / batch_size))
        batch_image_shape = [batch_size] + images_shape[1:]
        sess.run(self.model.init_op, {self.model.input_placeholder:np.zeros(batch_image_shape)})
        self.model.load_full_model(sess, self.analysis_params.cp_loc)
        activations = []
        for batch_idx in range(num_batches):
          im_batch_start_idx = int(batch_idx * batch_size)
          im_batch_end_idx = int(np.min([im_batch_start_idx + batch_size, num_images]))
          batch_images = images[im_batch_start_idx:im_batch_end_idx, ...]
          feed_dict = self.model.get_feed_dict(batch_images, is_test=True)
          outputs = sess.run(activation_operation(), feed_dict)
          activations.append(outputs.copy())
        activations = np.stack(activations, axis=0)
        num_batches, batch_size, num_outputs = activations.shape
        activations = activations.reshape((num_batches*batch_size, num_outputs))
      else:
        feed_dict = self.model.get_feed_dict(images, is_test=True)
        sess.run(self.model.init_op, feed_dict)
        self.model.load_full_model(sess, self.analysis_params.cp_loc)
        activations = sess.run(activation_operation(), feed_dict)
    return activations

  def compute_atas(self, activities, images, batch_size=100):
    """
    Returns activity triggered averages
    Outputs:
      atas [np.ndarray] of shape (num_pixels, num_neurons)
    Inputs:
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    num_images, num_pixels =  images.shape
    num_act_images, num_neurons = activities.shape
    assert num_act_images == num_images, (
      "activities.shape[0] = %g and images.shape[0] = %g must be equal"%(
      num_act_images, num_images))
    if num_images < batch_size: # No need to do batches
      return images.T.dot(activities) / num_images
    num_extra_images = num_images % batch_size
    num_batches = (num_images - num_extra_images) // batch_size
    atas = np.zeros([num_pixels, num_neurons])
    num_images_processed = 0
    for batch_index in range(num_batches):
      img_slice = images[num_images_processed:num_images_processed+batch_size, ...]
      act_slice = activities[num_images_processed:num_images_processed+batch_size, ...]
      atas +=  img_slice.T.dot(act_slice) / batch_size
      num_images_processed += batch_size
    atas /= num_batches
    if num_extra_images > 0: # there are still images left
      img_slice = images[-num_extra_images:, ...]
      act_slice = activities[-num_extra_images:, ...]
      batch_multiplier = num_images_processed / num_images
      atas = ((batch_multiplier * atas)
        + ((1 - batch_multiplier) * (img_slice.T.dot(act_slice) / num_extra_images)))
      num_images_processed += num_extra_images
    return atas

  def compute_atcs(self, activities, images, atas=None):
    """
    Returns activity triggered covariances
    Outputs:
      atcs [np.ndarray] of shape (num_neurons, num_pixels, num_pixels)
    Inputs:
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      atas [np.adarray] of shape (pixels, neurons) of pre-computed act trig avgs
    TODO: is it possible to vectorize this so it is faster?
    """
    num_batch, num_neurons = activities.shape
    images -= np.mean(images)
    if atas is None:
      atas = self.compute_atas(activities, images)
    atcs = [None,]*num_neurons
    for neuron_id in range(num_neurons):
      ata = atas[:, neuron_id] # pixels x 1
      img_deviations = images - ata[None, :] # batch x pixels
      img_vects = [img[None,:] for img in img_deviations]
      covs = [x.T.dot(x) for x in img_vects] # pixels x pixels x batch
      covs_acts = zip(covs, activities[:, neuron_id]) # (PxPxB, B)
      atcs[neuron_id] = sum([cov*act for cov,act in covs_acts]) / num_batch
    atcs = np.stack(atcs, axis=0)
    return atcs

  def grating_analysis(self, weight_stats, save_info):
    ot_grating_responses = self.orientation_tuning(weight_stats, self.analysis_params.contrasts,
      self.analysis_params.orientations, self.analysis_params.phases,
      self.analysis_params.neuron_indices, scale=self.analysis_params.input_scale)
    np.savez(self.analysis_out_dir+"savefiles/ot_responses_"+save_info+".npz",
      data=ot_grating_responses)
    ot_mean_activations = ot_grating_responses["mean_responses"]
    base_orientations = [
      self.analysis_params.orientations[np.argmax(ot_mean_activations[bf_idx, -1, :])]
      for bf_idx in range(len(ot_grating_responses["neuron_indices"]))]
    co_grating_responses = self.cross_orientation_suppression(self.bf_stats,
      self.analysis_params.contrasts, self.analysis_params.phases, base_orientations,
      self.analysis_params.orientations, self.analysis_params.neuron_indices,
      scale=self.analysis_params.input_scale)
    np.savez(self.analysis_out_dir+"savefiles/co_responses_"+save_info+".npz",
      data=co_grating_responses)
    self.analysis_logger.log_info("Grating  analysis is complete.")
    return (ot_grating_responses, co_grating_responses)

  def orientation_tuning(self, bf_stats, contrasts=[0.5], orientations=[np.pi],
    phases=[np.pi], neuron_indices=None, diameter=-1, scale=1.0):
    """
    Performs orientation tuning analysis for given parameters
    Inputs:
      bf_stats [dict] computed from utils/data_processing.get_dictionary_stats()
      contrasts [list or np.array] all experiments will be run at each contrast
      orientations [list or np.array] code will compute neuron response for each orientation
      phases [list or np.array] the mean and max response will be computed across all phases
      neuron_indices [list or np.array] the experiments will be run for each neuron index specified
        Setting this to the default, None, will result in performing the experiment on all neurons
      diameter [int] diameter of mask for grating stimulus, -1 indicates full-field stimulus
      scale [float] scale of the stimulus. By default the stimulus is scaled between -1 and 1.
    """
    # Stimulus parameters
    tot_num_bfs = bf_stats["num_outputs"]
    if neuron_indices is None:
      neuron_indices = np.arange(tot_num_bfs)
    num_pixels = bf_stats["patch_edge_size"]**2
    num_neurons = np.asarray(neuron_indices).size
    num_contrasts = np.asarray(contrasts).size
    num_orientations = np.asarray(orientations).size
    num_phases = np.asarray(phases).size
    # Generate a grating with spatial frequency provided by bf_stats
    grating = lambda neuron_idx,contrast,orientation,phase:dp.generate_grating(
      *dp.get_grating_params(bf_stats, neuron_idx, orientation=orientation,
      phase=phase, contrast=contrast, diameter=diameter))
    # Output arrays
    max_responses = np.zeros((num_neurons, num_contrasts, num_orientations))
    mean_responses = np.zeros((num_neurons, num_contrasts, num_orientations))
    rect_responses = np.zeros((num_neurons, 2, num_contrasts, num_orientations))
    best_phases = np.zeros((num_neurons, num_contrasts, num_orientations))
    phase_stims = np.stack([grating(neuron_idx, contrast, orientation, phase)
      for neuron_idx in neuron_indices
      for contrast in contrasts
      for orientation in orientations
      for phase in phases], axis=0) #Array containing all stimulus that can be returned for testing
    phase_stims = {"test": Dataset(phase_stims[:,:,:,None], lbls=None, ignore_lbls=None,
      rand_state=self.rand_state)}
    if not self.model_params.whiten_data:
      self.model_params.whiten_method = None
    phase_stims = self.model.preprocess_dataset(phase_stims,
      params={"whiten_data":self.model_params.whiten_data,
      "whiten_method":self.model_params.whiten_method})
    phase_stims = self.model.reshape_dataset(phase_stims, self.model_params)
    phase_stims["test"].images /= np.max(np.abs(phase_stims["test"].images))
    phase_stims["test"].images *= scale

    # compute_activations will give orientation tuning for whatever outputs are returned model.a
    activations = self.compute_activations(phase_stims["test"].images, batch_size=1024)
    activations = activations.reshape((
      num_neurons, num_contrasts, num_orientations, num_phases, tot_num_bfs))
    # If you're running a deep network and want orientation tuning for the first layer:
    # TODO: Make layer number a parameter & default in 'compute_activations'
    #var_name = self.model.module.u_list[1].name
    #activations = self.evaluate_model(phase_stims["test"].images,
    #  [var_name])[var_name].reshape(num_neurons, num_contrasts, num_orientations, num_phases,
    #  tot_num_bfs)

    for bf_idx, neuron_idx in enumerate(neuron_indices):
      activity_slice = activations[bf_idx, :, :, :, neuron_idx]
      max_responses[bf_idx, ...] = np.max(np.abs(activity_slice), axis=-1)
      mean_responses[bf_idx, ...] = np.mean(np.abs(activity_slice), axis=-1)
      rect_responses[bf_idx, 0, ...] = np.mean(np.maximum(0, activity_slice), axis=-1)
      rect_responses[bf_idx, 1, ...] = np.mean(np.maximum(0, -activity_slice), axis=-1)
      for co_idx, contrast in enumerate(contrasts):
        for or_idx, orientation in enumerate(orientations):
          phase_activity = activations[bf_idx, co_idx, or_idx, :, neuron_idx]
          best_phases[bf_idx, co_idx, or_idx] = phases[np.argmax(phase_activity)]
    return {"contrasts":contrasts, "orientations":orientations, "phases":phases,
      "neuron_indices":neuron_indices, "max_responses":max_responses,
      "mean_responses":mean_responses, "rectified_responses":rect_responses,
      "best_phases":best_phases}

  def cross_orientation_suppression(self, bf_stats, contrasts=[0.5], phases=[np.pi],
    base_orientations=[np.pi], mask_orientations=[np.pi/2], neuron_indices=None, diameter=-1,
    scale=1.0):
    """
    Performs orientation tuning analysis for given parameters
    Inputs:
      bf_stats [dict] computed from utils/data_processing.get_dictionary_stats()
      contrasts [list or np.array] all experiments will be run at each contrast
      phases [list or np.array] the mean and max response will be computed across all phases
      base_orientations [list or np.array] each neuron will have a unique base orientation
        should be the same len as neuron_indices
      mask_orientations [list or np.array] will compute neuron response for each mask_orientation
      neuron_indices [list or np.array] the experiments will be run for each neuron index specified
        Setting this to the default, None, will result in performing the experiment on all neurons
      diameter [int] diameter of mask for grating stimulus, -1 indicates full-field stimulus
      scale [float] scale of the stimulus. By default the stimulus is scaled between -1 and 1.
    """
    # Stimulus parameters
    tot_num_bfs = bf_stats["num_outputs"]
    if neuron_indices is None:
      neuron_indices = np.arange(tot_num_bfs)
    num_contrasts = np.asarray(contrasts).size
    num_phases = np.asarray(phases).size
    num_orientations = np.asarray(mask_orientations).size
    num_neurons = np.asarray(neuron_indices).size
    num_pixels = bf_stats["patch_edge_size"]**2
    assert np.asarray(base_orientations).size == num_neurons, (
      "You must specify a base orientation for all basis functions")
    # Generate a grating with spatial frequency estimated from bf_stats
    grating = lambda neuron_idx,contrast,orientation,phase:dp.generate_grating(
      *dp.get_grating_params(bf_stats, neuron_idx, orientation=orientation,
      phase=phase, contrast=contrast, diameter=diameter))
    # Output arrays
    base_max_responses = np.zeros((num_neurons, num_contrasts))
    test_max_responses = np.zeros((num_neurons, num_contrasts, num_contrasts, num_orientations))
    base_mean_responses = np.zeros((num_neurons, num_contrasts))
    test_mean_responses = np.zeros((num_neurons, num_contrasts, num_contrasts, num_orientations))
    base_rect_responses = np.zeros((num_neurons, 2, num_contrasts))
    test_rect_responses = np.zeros((num_neurons, 2, num_contrasts, num_contrasts, num_orientations))
    for bf_idx, neuron_idx in enumerate(neuron_indices): # each neuron produces a base & test output
      for bco_idx, base_contrast in enumerate(contrasts): # loop over base contrast levels
        base_stims = np.stack([grating(neuron_idx, base_contrast, base_orientations[bf_idx],
          base_phase) for base_phase in phases], axis=0)
        base_stims = {"test": Dataset(base_stims[:,:,:,None], lbls=None, ignore_lbls=None,
          rand_state=self.rand_state)}
        base_stims = self.model.preprocess_dataset(base_stims,
          params={"whiten_data":self.model_params.whiten_data,
          "whiten_method":self.model_params.whiten_method})
        base_stims = self.model.reshape_dataset(base_stims, self.model_params)
        base_stims["test"].images /= np.max(np.abs(base_stims["test"].images))
        base_stims["test"].images *= scale

        base_activity = self.compute_activations(base_stims["test"].images)[:, neuron_idx]
        #TODO: param for this (see above TODO)
        #var_name = self.model.module.u_list[1].name
        #base_activity = self.evaluate_model(base_stims["test"].images,
        #  [var_name])[var_name][:, neuron_idx]

        base_max_responses[bf_idx, bco_idx] = np.max(np.abs(base_activity))
        base_mean_responses[bf_idx, bco_idx] = np.mean(np.abs(base_activity))
        base_rect_responses[bf_idx, 0, bco_idx] = np.mean(np.maximum(0, base_activity))
        base_rect_responses[bf_idx, 1, bco_idx] = np.mean(np.maximum(0, -base_activity))
        # generate mask stimulus to overlay onto base stimulus
        for co_idx, mask_contrast in enumerate(contrasts): # loop over test contrasts
            test_stims = np.zeros((num_orientations, num_phases, num_phases, num_pixels))
            for or_idx, mask_orientation in enumerate(mask_orientations):
              mask_stims = np.stack([grating(neuron_idx, mask_contrast, mask_orientation, mask_phase)
                for mask_phase in phases], axis=0)
              mask_stims = {"test": Dataset(mask_stims[:,:,:,None], lbls=None, ignore_lbls=None,
                rand_state=self.rand_state)}
              mask_stims = self.model.preprocess_dataset(mask_stims,
                params={"whiten_data":self.model_params.whiten_data,
                "whiten_method":self.model_params.whiten_method})
              mask_stims = self.model.reshape_dataset(mask_stims, self.model_params)
              mask_stims["test"].images /= np.max(np.abs(mask_stims["test"].images))
              mask_stims["test"].images *= scale
              test_stims[or_idx, ...] = base_stims["test"].images[:,None,:] + mask_stims["test"].images[None,:,:]
            test_stims = test_stims.reshape(num_orientations*num_phases*num_phases, num_pixels)

            test_activity = self.compute_activations(test_stims)[:, neuron_idx]
            #TODO: param for this (see above TODO)
            #var_name = self.model.module.u_list[1].name
            #test_activity = self.evaluate_model(test_stims,
            #  [var_name])[var_name][:, neuron_idx]

            test_activity = np.reshape(test_activity, (num_orientations, num_phases**2))
            # peak-to-trough amplitude is computed across all base & mask phases
            test_max_responses[bf_idx, bco_idx, co_idx, :] = np.max(np.abs(test_activity), axis=1)
            test_mean_responses[bf_idx, bco_idx, co_idx, :] = np.mean(np.abs(test_activity), axis=1)
            test_rect_responses[bf_idx, 0, bco_idx, co_idx, :] = np.mean(np.maximum(0,
              test_activity), axis=1)
            test_rect_responses[bf_idx, 1, bco_idx, co_idx, :] = np.mean(np.maximum(0,
              -test_activity), axis=1)
    return {"contrasts":contrasts, "phases":phases, "base_orientations":base_orientations,
      "neuron_indices":neuron_indices, "mask_orientations":mask_orientations,
      "base_max_responses":base_max_responses, "test_max_responses":test_max_responses,
      "base_mean_responses":base_mean_responses,  "test_mean_responses":test_mean_responses,
      "base_rect_responses":base_rect_responses, "test_rect_responses":test_rect_responses}

  def iso_response_contrasts(self, bf_stats, base_contrast=0.5, contrast_resolution=0.01,
    closeness=0.01, num_alt_orientations=1, orientations=[np.pi, np.pi/2], phases=[np.pi],
    neuron_indices=None, diameter=-1, scale=1.0):
    """
    Computes what contrast has maximally close response to the response at 0.5 contrast for a given
    orientation
    Inputs:
      bf_stats [dict] computed from utils/data_processing.get_dictionary_stats()
      contrast_resoultion [float] how much to increment contrast when searching for equal response
      closeness [float] what resolution constitutes a close response (atol in np.isclose)
      num_alt_orientations [int] how many other orientations to test
        must be < num_orientations
      orientations [list or np.array] code will compute neuron response for each orientation
        len(orientations) must be > 1
      phases [list or np.array] the mean and max response will be computed across all phases
      neuron_indices [list or np.array] the experiments will be run for each neuron index specified
        Setting this to the default, None, will result in performing the experiment on all neurons
      diameter [int] diameter of mask for grating stimulus, -1 indicates full-field stimulus
      scale [float] scale of the stimulus. By default the stimulus is scaled between -1 and 1.
    """
    # Stimulus parameters
    tot_num_bfs = bf_stats["num_outputs"]
    if neuron_indices is None:
      neuron_indices = np.arange(tot_num_bfs)
    num_neurons = np.asarray(neuron_indices).size
    num_pixels = bf_stats["patch_edge_size"]**2
    num_orientations = np.asarray(orientations).size
    assert num_alt_orientations < num_orientations, (
      "num_alt_orientations must be < num_orientations")
    num_phases = np.asarray(phases).size
    # Generate a grating with spatial frequency provided by bf_stats
    grating = lambda neuron_idx,contrast,orientation,phase:dp.generate_grating(
      *dp.get_grating_params(bf_stats, neuron_idx, orientation=orientation,
      phase=phase, contrast=contrast, diameter=diameter)).reshape(num_pixels)
    # Output arrays
    iso_response_parameters = []
    # Stimulus
    raw_phase_stims = np.stack([grating(neuron_idx, base_contrast, orientation, phase)
      for neuron_idx in neuron_indices
      for orientation in orientations
      for phase in phases], axis=0) #Array containing all stimulus that can be returned for testing
    if hasattr(self.model_params, "whiten_data") and self.model_params.whiten_data:
      phase_stims, phase_mean, phase_filter = dp.whiten_data(raw_phase_stims,
        method=self.model_params.whiten_method)
    if hasattr(self.model_params, "lpf_data") and self.model_params.lpf_data:
      phase_stims, phase_mean, phase_filter = dp.lpf_data(raw_phase_stims,
        cutoff=self.model_params.lpf_cutoff)
    phase_stims = scale * (phase_stims / np.max(np.abs(phase_stims)))
    activations = self.compute_activations(phase_stims).reshape(num_neurons, num_orientations,
      num_phases, tot_num_bfs)
    for bf_idx, neuron_idx in enumerate(neuron_indices):
      activity_slice = activations[bf_idx, :, :, neuron_idx] #[orientation, phase]
      mean_responses = np.mean(np.abs(activity_slice), axis=-1) # mean across phase
      best_or_idx = np.argmax(mean_responses)
      best_ph_idx = np.argmax(activity_slice[best_or_idx, :])
      target_response = mean_responses[best_or_idx]
      bf_iso_response_parameters = [(base_contrast, orientations[best_or_idx],
        phases[best_ph_idx], target_response)]
      left_orientations = orientations[:best_or_idx][::-1][:int(np.floor(num_alt_orientations/2))]
      right_orientations = orientations[best_or_idx:][:int(np.ceil(num_alt_orientations/2))]
      alt_orientations = list(left_orientations)+list(right_orientations)
      for orientation in alt_orientations:
        response = 0.0
        contrast = base_contrast
        loop_exit = False
        prev_contrast = base_contrast
        while not loop_exit:
          if contrast <= 1.0:
            raw_test_stims = np.stack([grating(neuron_idx, contrast, orientation, phase)
              for phase in phases], axis=0)
            if hasattr(self.model_params, "whiten_data") and self.model_params.whiten_data:
              test_stims, test_mean, test_filter = dp.whiten_data(raw_test_stims,
                method=self.model_params.whiten_method)
            if hasattr(self.model_params, "lpf_data") and self.model_params.lpf_data:
              test_stims, test_mean, test_filter = dp.lpf_data(raw_test_stims,
                cutoff = self.model_params.lpf_cutoff)
            test_stims = scale * (test_stims / np.max(np.abs(test_stims)))
            bf_activations = self.compute_activations(test_stims).reshape(num_phases,
              tot_num_bfs)[:, neuron_idx]
            best_phase = phases[np.argmax(bf_activations)]
            response = np.mean(np.abs(bf_activations))
            if response >= target_response:
              if response == target_response:
                target_found = True
                target_contrast = contrast
              else:
                target_found = True
                target_contrast = prev_contrast - (prev_contrast - contrast)/2.0
              prev_contrast = contrast
            else:
              contrast += contrast_resolution
              target_found = False
            loop_exit = target_found # exit if you found a target
          else:
            target_found = False
            loop_exit = True
        if target_found:
         bf_iso_response_parameters.append((target_contrast, orientation, best_phase, response))
      iso_response_parameters.append(bf_iso_response_parameters)
    return {"orientations":orientations, "phases":phases, "neuron_indices":neuron_indices,
      "iso_response_parameters":iso_response_parameters}

  def run_patch_recon_analysis(self, full_image, save_info):
    """
    Break image into patches, compute recons, reassemble recons back into a full image
    """
    self.full_image = full_image
    if self.model_params.whiten_data:
      # FT method is the only one that works on full images
      wht_img, img_mean, ft_filter = dp.whiten_data(full_image,
        method="FT", lpf_cutoff=self.model_params.lpf_cutoff)
    else:
      wht_img = full_image
    img_patches = dp.extract_patches(wht_img,
      out_shape=(1, self.model_params.patch_edge_size, self.model_params.patch_edge_size, 1),
      overlapping=False, randomize=False, var_thresh=0.0)
    img_patches, orig_shape = dp.reshape_data(img_patches, flatten=True)[:2]
    model_eval = self.evaluate_model(img_patches,
      ["inference/activity:0", "output/reconstruction:0"])
    recon_patches = model_eval["output/reconstruction:0"]
    a_vals = model_eval["inference/activity:0"]
    self.recon_frac_act = np.array(np.count_nonzero(a_vals) / float(a_vals.size))
    recon_patches = dp.reshape_data(recon_patches, flatten=False, out_shape=orig_shape)[0]
    self.full_recon = dp.patches_to_image(recon_patches, full_image.shape).astype(np.float32)
    if self.model_params.whiten_data:
      self.full_recon = dp.unwhiten_data(self.full_recon, img_mean, ft_filter, method="FT")
    np.savez(self.analysis_out_dir+"savefiles/full_recon_"+save_info+".npz",
      data={"full_image":self.full_image, "full_recon":self.full_recon,
      "recon_frac_act":self.recon_frac_act})
    self.analysis_logger.log_info("Patch recon analysis is complete.")

  def get_neuron_angles(self, bf_stats):
    """
    Compute the angle in degrees between all pairs of basis functions in bf_stats
    Outputs:
      neuron_angles [np.ndarray] lower triangle of plot matrix only, as a vector in raster order
      plot_matrix [np.ndarray] of shape [num_neurons, num_neurons] with all angles between
        basis functions in the lower triangle and upper triangle is set to -1
    Inputs:
      bf_stats [dict] returned from utils/data_processing.get_dictionary_stats()
    """
    num_pixels = bf_stats["patch_edge_size"]**2
    indices = np.tril_indices(bf_stats["num_outputs"], 1)
    vect_size = len(indices[0])
    neuron_angles = np.zeros(vect_size)
    plot_matrix = np.zeros((bf_stats["num_outputs"], bf_stats["num_outputs"]))
    for angleid, (nid0, nid1) in enumerate(zip(*indices)):
      bf0 = bf_stats["basis_functions"][nid0].reshape((num_pixels, 1))
      bf1 = bf_stats["basis_functions"][nid1].reshape((num_pixels, 1))
      inner_products = np.dot((bf0 / np.linalg.norm(bf0)).T, (bf1 / np.linalg.norm(bf1)))
      inner_products[inner_products>1.0] = 1.0
      inner_products[inner_products<-1.0] = -1.0
      angle = np.arccos(inner_products)
      neuron_angles[angleid] = angle * (180/np.pi)
      plot_matrix[nid0, nid1] = angle * (180/np.pi)
    plot_matrix[plot_matrix==0] = -1
    return neuron_angles, plot_matrix

  def bf_projections(self, bf0, bf1):
    return dp.bf_projections(bf0, bf1)

  def neuron_visualization_analysis(self, save_info=""):
    """
    TODO: docstrings; incl data preprocessing pipeline?
    """
    vis_data_init = np.random.normal(loc=0.0, scale=1e-2, size=self.model.get_input_shape()[1:])
    vis_data_init /= np.linalg.norm(vis_data_init)
    vis_data_init = vis_data_init[None,:]
    self.neuron_vis_output = {
      "data_init":vis_data_init,
      "steps":[],
      "optimal_stims":[],
      "loss":[]}
    for target_neuron_idx in self.analysis_params.neuron_vis_targets:
      selection_vector = np.zeros(self.model.get_num_latent())
      selection_vector[target_neuron_idx] = 1
      optimal_stim_outputs = self.construct_optimal_stimulus(vis_data_init, selection_vector)
      self.neuron_vis_output["steps"].append(optimal_stim_outputs["steps"])
      self.neuron_vis_output["optimal_stims"].append(optimal_stim_outputs["images"])
      self.neuron_vis_output["loss"].append(optimal_stim_outputs["loss"])
    np.savez(self.analysis_out_dir+"savefiles/neuron_visualization_analysis_"+save_info+".npz",
      data=self.neuron_vis_output)
    self.analysis_logger.log_info("Neuron visualization analysis is complete.")

  def construct_optimal_stimulus(self, init_image, selection_vector):
    """
    Constructs optimal stimulus for a given neuron
    Inputs:
      init_image: [np.ndarray] image to initialize optimal search with
    Outputs:
      out_dict: [dictionary] with keys
        "steps" - step number for each output
        "images" - optimal stimulus image at given step
        "loss" - visualization loss at given step
    """
    input_shape = self.model.get_input_shape()[1:] # We don't need bach dim
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(init_image, is_test=True)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      out_dict = self.neuron_vis_module.construct_optimal_stimulus(feed_dict,
        selection_vector=selection_vector,
        stim_save_int=self.analysis_params.neuron_vis_stim_save_int,
        save_int=self.analysis_params.neuron_vis_save_int)
    return out_dict

  def construct_recon_adversarial_stimulus(self, input_images, target_images):
    if(self.analysis_params.adversarial_attack_method == "kurakin_targeted" or
      self.analysis_params.adversarial_attack_method == "marzi_untargeted" or
      self.analysis_params.adversarial_attack_method == "marzi_latent"):
      #Not using recon_mult here, so set arbitrary value
      self.analysis_params.carlini_recon_mult = [0]
    elif(self.analysis_params.adversarial_attack_method == "carlini_targeted"):
      if(type(self.analysis_params.carlini_recon_mult) is not list):
        self.analysis_params.carlini_recon_mult = [self.analysis_params.carlini_recon_mult]
    else:
      assert False, (
        "Adversarial attack method must be"
        +"'kurakin_targeted', 'carlini_targeted', 'marzi_untargted', or 'marzi_latent'.")
    input_target_mse = dp.mse(input_images, target_images)
    distances = {"input_target_mse":input_target_mse, "input_recon_mses":[],
    "input_adv_mses":[], "target_recon_mses":[],
    "target_adv_mses":[], "adv_recon_mses":[], "target_adv_cos_similarities":[],
    "input_adv_cos_similarities":[], "target_pert_cos_similarities": [],
    "input_pert_cos_similarities":[], "adv_loss":[]}
    steps=None
    all_adversarial_images = []
    all_recons = []
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(input_images, is_test=True)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      for r_mult in self.analysis_params.carlini_recon_mult:
        out_dict = self.recon_adv_module.construct_adversarial_examples(feed_dict,
          recon_mult=r_mult,
          target_generation_method="specified",
          target_images=target_images,
          selection_vector=self.analysis_params.neuron_vis_selection_vector,
          save_int=self.analysis_params.adversarial_save_int)
        steps = out_dict["step"]
        distances["adv_loss"].append(out_dict["adv_loss"])
        distances["input_recon_mses"].append(out_dict["input_recon_mses"])
        distances["input_adv_mses"].append(out_dict["input_adv_mses"])
        distances["target_recon_mses"].append(out_dict["target_recon_mses"])
        distances["target_adv_mses"].append(out_dict["target_adv_mses"])
        distances["adv_recon_mses"].append(out_dict["adv_recon_mses"])
        distances["target_adv_cos_similarities"].append(out_dict["target_adv_sim"])
        distances["input_adv_cos_similarities"].append(out_dict["input_adv_sim"])
        distances["target_pert_cos_similarities"].append(out_dict["target_pert_sim"])
        distances["input_pert_cos_similarities"].append(out_dict["input_pert_sim"])
        all_adversarial_images.append(out_dict["adv_images"])
        all_recons.append(out_dict["adv_recons"])
    return steps, all_adversarial_images, all_recons, distances

  def recon_adversary_analysis(self, images, labels=None, batch_size=1, input_id=None,
    target_method="random", target_id=None, save_info=""):
    #Default parameters
    if input_id is None:
      input_id = np.arange(images.shape[0]).astype(np.int32)
    else:
      input_id = np.array(input_id)
    self.num_data = input_id.shape[0]
    #If batch_size is None, do all in one batch
    if batch_size is None:
      batch_size = self.num_data
    input_images = images[input_id, ...].astype(np.float32)
    num_images = images.shape[0]
    #Define target label based on target method
    if(target_method == "random"):
      target_id = input_id.copy()
      #If labels is defined, resample until target label is not true label
      if(labels is not None):
        input_labels = labels[input_id, ...].astype(np.float32)
        input_classes = np.argmax(input_labels, axis=-1)
        target_classes = input_classes.copy()
        while(np.any(target_classes == input_classes)):
          resample_idx = np.nonzero(target_classes == input_classes)
          target_id[resample_idx] = self.rand_state.randint(
            0, num_images, size=resample_idx[0].shape)
          target_labels = labels[target_id, ...].astype(np.float32)
          target_classes = np.argmax(target_labels, axis=-1)
      else:
        #Resample until target_id is not input_id
        #Also check labels if set
        while(np.any(target_id == input_id)):
          resample_idx = np.nonzero(target_id == input_id)
          target_id[resample_idx] = self.rand_state.randint(
            0, num_images, size=resample_idx[0].shape)
    elif(target_method == "specified"):
      assert(target_id is not None)
      target_id = np.array(target_id)
      assert(target_id.shape[0] == self.num_data)
    else:
      assert False, ("Allowed target methods for recon adversary are " +
        "\"random\" or \"specified\"")
    if(self.analysis_params.adversarial_attack_method == "kurakin_targeted"):
      num_recon_mults = 1
    elif(self.analysis_params.adversarial_attack_method == "carlini_targeted"):
      num_recon_mults = len(self.analysis_params.carlini_recon_mult)
    else:
      assert False
    #Make sure that the save interval is less than num steps, otherwise
    #it won't store the adv exmaples
    assert self.analysis_params.adversarial_save_int <= self.analysis_params.adversarial_num_steps,\
      ("Save interval must be <= adversarial_num_steps")
    num_stored_steps = ((self.analysis_params.adversarial_num_steps)//self.analysis_params.adversarial_save_int) + 1
    self.adversarial_input_target_mses = np.zeros((self.num_data))
    self.adversarial_input_recon_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    self.adversarial_input_adv_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    self.adversarial_target_recon_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    self.adversarial_target_adv_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    self.adversarial_adv_recon_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    self.adversarial_images = np.zeros((num_recon_mults, num_stored_steps,) + input_images.shape)
    self.adversarial_recons = np.zeros((num_recon_mults, num_stored_steps,) + input_images.shape)
    num_iterations = int(np.ceil(self.num_data / batch_size))
    target_images = images[target_id, ...].astype(np.float32)
    for it in range(num_iterations):
      batch_start_idx = int(it * batch_size)
      batch_end_idx = int(np.min([batch_start_idx + batch_size, self.num_data]))
      batch_input_images = input_images[batch_start_idx:batch_end_idx, ...]
      batch_target_images = target_images[batch_start_idx:batch_end_idx, ...]
      self.steps_idx, batch_adv_images, batch_adv_recons, distances = \
        self.construct_recon_adversarial_stimulus(batch_input_images, batch_target_images)
      #Store output variables
      self.adversarial_input_target_mses[batch_start_idx:batch_end_idx] = \
        np.array(distances["input_target_mse"])
      self.adversarial_input_recon_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(distances["input_recon_mses"])
      self.adversarial_input_adv_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(distances["input_adv_mses"])
      self.adversarial_target_recon_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(distances["target_recon_mses"])
      self.adversarial_target_adv_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(distances["target_adv_mses"])
      self.adversarial_adv_recon_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(distances["adv_recon_mses"])
      self.adversarial_images[:, :, batch_start_idx:batch_end_idx, ...] = \
        np.array(batch_adv_images)
      self.adversarial_recons[:, :, batch_start_idx:batch_end_idx, ...] = \
        np.array(batch_adv_recons)
    self.recon_adversarial_input_images = input_images
    self.adversarial_target_images = target_images
    #Store everything in out dictionaries
    out_dicts = [{}, {}]
    out_dicts[0]["steps_idx"] = self.steps_idx
    out_dicts[0]["input_images"] = input_images
    out_dicts[0]["target_images"] = target_images
    out_dicts[0]["adversarial_images"] = self.adversarial_images
    out_dicts[0]["num_data"] = self.num_data
    out_dicts[0]["step_size"] = self.analysis_params.adversarial_step_size
    out_dicts[0]["num_steps"] = self.analysis_params.adversarial_num_steps
    out_dicts[0]["input_id"] = input_id
    out_dicts[0]["target_id"] = target_id
    out_dicts[0].update(distances)
    out_dicts[1]["adversarial_recons"] = self.adversarial_recons
    np.savez(self.analysis_out_dir+"savefiles/recon_adversary_stats_"+save_info+".npz",
      data=out_dicts[0])
    np.savez(self.analysis_out_dir+"savefiles/recon_adversary_recons_"+save_info+".npz",
      data=out_dicts[1])
    self.analysis_logger.log_info("Adversary analysis is complete.")


  def construct_class_adversarial_stimulus(self, input_images, input_labels,
    target_labels):
    if(self.analysis_params.adversarial_attack_method == "kurakin_untargeted"):
      assert(target_labels is not None)
      #Not using recon_mult here, so set arb value
      self.analysis_params.carlini_recon_mult = [0]
    elif(self.analysis_params.adversarial_attack_method == "kurakin_targeted"):
      #Not using recon_mult here, so set arb value
      self.analysis_params.carlini_recon_mult = [0]
    elif(self.analysis_params.adversarial_attack_method == "carlini_targeted"):
      assert(target_labels is not None)
      if(type(self.analysis_params.carlini_recon_mult) is not list):
        self.analysis_params.carlini_recon_mult = [self.analysis_params.carlini_recon_mult]
    else:
      assert False, ("Adversarial attack method must be "+\
        "\"kurakin_untargeted\", \"kurakin_targeted\", or \"carlini_targeted\"")
    mses = {"input_adv_mses":[], "target_output_losses":[]}
    steps = None
    all_adv_images = []
    all_adv_outputs = []
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(input_images, input_labels, is_test=True)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      for r_mult in self.analysis_params.carlini_recon_mult:
        out_dict = self.class_adv_module.construct_adversarial_examples(
          feed_dict,
          labels=input_labels,
          recon_mult=r_mult,
          rand_state=self.rand_state,
          target_generation_method="specified",
          target_labels=target_labels,
          save_int=self.analysis_params.adversarial_save_int)
        steps = out_dict["step"]
        all_adv_images.append(out_dict["adv_images"])
        all_adv_outputs.append(out_dict["adv_outputs"])
        mses["input_adv_mses"].append(out_dict["input_adv_mses"])
        mses["target_output_losses"].append(out_dict["adv_losses"])
    return steps, all_adv_images, all_adv_outputs, mses

  def get_adv_indices(self, softmax_conf, all_kept_indices, confidence_threshold, num_images, labels):
    softmax_conf[np.arange(num_images, dtype=np.int32), labels] = 0 # zero confidence at true label
    confidence_indices = np.max(softmax_conf, axis=-1) # highest non-true label confidence
    adversarial_labels = np.argmax(softmax_conf, axis=-1) # index of highest non-true label
    all_above_thresh = np.nonzero(np.squeeze(confidence_indices>confidence_threshold))[0]
    keep_indices = np.array([], dtype=np.int32)
    for adv_index in all_above_thresh:
      if adv_index not in set(all_kept_indices):
        keep_indices = np.append(keep_indices, adv_index)
    return keep_indices, confidence_indices, adversarial_labels

  def class_adversary_analysis(self, images, labels, batch_size=1, input_id=None,
      target_method="random", target_labels=None, save_info=""):

    assert(images.shape[0] == labels.shape[0])

    #Default parameters
    #TODO need to make sure these are getting correct classifications
    #if input_id is None, use all avaliable data
    if input_id is None:
      input_id = np.arange(images.shape[0]).astype(np.int32)
    else:
      input_id = np.array(input_id)

    self.num_data = input_id.shape[0]
    #If batch_size is None, do all in one batch
    if batch_size is None:
      batch_size = self.num_data

    input_images = images[input_id, ...].astype(np.float32)
    input_labels = labels[input_id, ...].astype(np.float32) # [num_inputs, num_classes] one-hot representation
    num_classes = input_labels.shape[-1]

    #Define target label based on target method
    if(target_method == "random"):
      input_classes = np.argmax(input_labels, axis=-1)
      target_labels = input_classes.copy()
      #Resample until target label is not a true label
      #TODO this is also defined in class_adv_module
      while(np.any(target_labels == input_classes)):
        resample_idx = np.nonzero(target_labels == input_classes)
        target_labels[resample_idx] = self.rand_state.randint(0, num_classes, size=resample_idx[0].shape)
    elif(target_method == "specified"):
      assert(target_labels is not None)
      target_labels = np.array(target_labels)
      assert(target_labels.shape[0] == self.num_data)
    else:
      assert False, ("Allowed target methods for classification adversary are " +
        "\"random\" or \"specified\"")

    #Check if target_labels is a class or one hot
    #If class, convert to one hot
    if(target_labels is not None):
      if(target_labels.ndim == 1):
        num_labels = target_labels.shape[0]
        out = np.zeros((num_labels, num_classes))
        #Convert target_labels into idx
        target_labels_idx = (np.arange(num_labels).astype(np.int32), target_labels)
        out[target_labels_idx] = 1
        target_labels = out

    if(self.analysis_params.adversarial_attack_method == "kurakin_untargeted" or
      self.analysis_params.adversarial_attack_method == "kurakin_targeted"):
      num_recon_mults = 1
    elif(self.analysis_params.adversarial_attack_method == "carlini_targeted"):
      num_recon_mults = len(self.analysis_params.carlini_recon_mult)
    else:
      assert False

    #Make sure that the save interval is less than num steps, otherwise
    #it won't store the adv exmaples
    assert self.analysis_params.adversarial_save_int <= self.analysis_params.adversarial_num_steps,  \
      ("Save interval must be <= adversarial_num_steps")

    #+1 since we always save the initial step
    num_stored_steps = ((self.analysis_params.adversarial_num_steps)//self.analysis_params.adversarial_save_int) + 1

    #TODO abstract this out into a "evaluate with batches" function
    #since vis_class_adv needs this as well

    #Output variables to store
    #In [num_recon_mults, num_steps, num_data]
    self.adversarial_input_adv_mses = np.zeros((num_recon_mults, num_stored_steps, self.num_data))
    #In [num_recon_mults, num_steps], summed over batches
    self.adversarial_target_output_losses = np.zeros((num_recon_mults, num_stored_steps))
    #In [num_recon_mults, num_steps, num_data, ny, nx, nf]
    self.adversarial_images = np.zeros((num_recon_mults, num_stored_steps,) + input_images.shape)
    #In [num_recon_mults, num_steps, num_data, num_classes]
    self.adversarial_outputs = np.zeros((num_recon_mults, num_stored_steps,) + input_labels.shape)

    #Split data into batches
    num_iterations = int(np.ceil(self.num_data / batch_size))

    for it in range(num_iterations):
      batch_start_idx = int(it * batch_size)
      batch_end_idx = int(np.min([batch_start_idx + batch_size, self.num_data]))
      batch_input_images = input_images[batch_start_idx:batch_end_idx, ...]
      batch_input_labels = input_labels[batch_start_idx:batch_end_idx, ...]
      if(target_labels is not None):
        batch_target_labels = target_labels[batch_start_idx:batch_end_idx, ...]
      else:
        batch_target_labels = None

      self.steps_idx, batch_adv_images, batch_adv_outputs, mses =  \
        self.construct_class_adversarial_stimulus(batch_input_images, batch_input_labels,
        batch_target_labels)

      #Store output variables
      self.adversarial_input_adv_mses[:, :, batch_start_idx:batch_end_idx] = \
        np.array(mses["input_adv_mses"])
      self.adversarial_target_output_losses[:, :] += np.array(mses["target_output_losses"])
      self.adversarial_images[:, :, batch_start_idx:batch_end_idx, ...] = \
        np.array(batch_adv_images)
      self.adversarial_outputs[:, :, batch_start_idx:batch_end_idx, ...] = \
        np.array(batch_adv_outputs)

    #Calculate total accuracies
    clean_est_classes = np.argmax(self.adversarial_outputs[:, 0, :, :], axis=-1)
    adv_est_classes = np.argmax(self.adversarial_outputs[:, -1, :, :], axis=-1)
    input_classes = np.argmax(input_labels, axis=-1)
    target_classes = np.argmax(target_labels, axis=-1)

    self.adversarial_clean_accuracy = np.mean(clean_est_classes == input_classes[None, ...], axis=-1)
    self.adversarial_adv_accuracy = np.mean(adv_est_classes == input_classes[None, ...], axis=-1)
    if(target_labels is not None):
      #Success rate is number of classifications targeted at target label
      self.adversarial_success_rate = np.mean(adv_est_classes == target_classes[None, ...], axis=-1)
    else:
      #Success rate is number of misclassifications (for untargeted attack)
      #TODO should this take into account clean accuracy?
      self.adversarial_success_rate = 1.0 - self.adversarial_adv_accuracy

    out_dicts = [{}, {}]
    #conf_based results
    if self.analysis_params.confidence_threshold > 0.0:
        store_data = np.zeros_like(images)
        store_time_step = -1*np.ones(images.shape[0], dtype=np.int32)
        store_labels = np.zeros(images.shape[0], dtype=np.int32)
        store_confidence = np.zeros(images.shape[0], dtype=np.float32)
        store_mses = np.zeros(images.shape[0], dtype=np.float32)
        all_kept_indices = []
        for adv_step in range(1, self.analysis_params.adversarial_num_steps+1): # first one is original
            keep_indices, confidence_indices, adversarial_labels = self.get_adv_indices(
                self.adversarial_outputs[0, adv_step, ...],
                all_kept_indices,
                self.analysis_params.confidence_threshold,
                images.shape[0],
                dp.one_hot_to_dense(labels.astype(np.int32)))
            if keep_indices.size > 0:
                all_kept_indices.extend(keep_indices)
                store_data[keep_indices, ...] = self.adversarial_images[0, adv_step, keep_indices, ...]
                store_time_step[keep_indices] = adv_step
                store_confidence[keep_indices] = confidence_indices[keep_indices]
                store_mses[keep_indices] = self.adversarial_input_adv_mses[0, adv_step, keep_indices]
                store_labels[keep_indices] = adversarial_labels[keep_indices]
        batch_indices = np.arange(images.shape[0], dtype=np.int32)[:,None]
        failed_indices = np.array([val for val in batch_indices if val not in all_kept_indices])
        if len(failed_indices) > 0:
            store_confidence[failed_indices] = confidence_indices[failed_indices]
            store_labels[failed_indices] = adversarial_labels[failed_indices]
            store_data[failed_indices, ...] = images[failed_indices, ...]
            store_mses[failed_indices] = self.adversarial_input_adv_mses[0, -1, failed_indices]
        self.conf_adversarial_images = [store_data]
        self.adversarial_time_step = [store_time_step]
        self.adversarial_confidence = [store_confidence]
        self.failed_indices = [failed_indices]
        self.success_indices = [list(set(all_kept_indices))]
        self.conf_adversarial_labels = [store_labels]
        self.mean_squared_distances = [store_mses]
        self.num_failed = [images.shape[0] - len(set(all_kept_indices))]
        out_dicts[0]['conf_adversarial_images'] = self.conf_adversarial_images
        out_dicts[0]['conf_adversarial_time_step'] = self.adversarial_time_step
        out_dicts[0]['conf_adversarial_confidence'] = self.adversarial_confidence
        out_dicts[0]['conf_failed_indices'] = self.failed_indices
        out_dicts[0]['conf_success_indices'] = self.success_indices
        out_dicts[0]['conf_adversarial_labels'] = self.conf_adversarial_labels
        out_dicts[0]['conf_mean_squared_distances'] = self.mean_squared_distances
        out_dicts[0]['conf_num_failed'] = self.num_failed

    #Store everything in out dictionaries
    out_dicts[0]['steps_idx'] = self.steps_idx
    out_dicts[0]['input_images'] = input_images
    out_dicts[0]['input_labels'] = input_labels
    out_dicts[0]['target_labels'] = target_labels
    out_dicts[0]['adversarial_outputs'] = self.adversarial_outputs
    out_dicts[0]['num_data'] = self.num_data
    out_dicts[0]['step_size'] = self.analysis_params.adversarial_step_size
    out_dicts[0]['num_steps'] = self.analysis_params.adversarial_num_steps
    out_dicts[0]['confidence_threshold'] = self.analysis_params.confidence_threshold
    out_dicts[0]['input_id'] = input_id
    out_dicts[0]['input_adv_mses'] = self.adversarial_input_adv_mses
    out_dicts[0]['target_output_losses'] = self.adversarial_target_output_losses
    out_dicts[0]['clean_accuracy'] = self.adversarial_clean_accuracy
    out_dicts[0]['adv_accuracy'] = self.adversarial_adv_accuracy
    out_dicts[0]['attack_success_rate'] = self.adversarial_success_rate
    np.savez(self.analysis_out_dir+'savefiles/class_adversary_'+save_info+'.npz',
      data=out_dicts[0])
    max_save = 50
    num_saved =  self.adversarial_images.shape[1]
    if num_saved > max_save:
      save_indices = np.arange(0, num_saved, num_saved//max_save)
      out_dicts[1]['adversarial_images'] = self.adversarial_images[:, save_indices, ...]
    else:
      out_dicts[1]['adversarial_images'] = self.adversarial_images
    try:
      np.savez(self.analysis_out_dir+'savefiles/class_adversary_images_'+save_info+'.npz',
        data=out_dicts[1])
      self.analysis_logger.log_info('Adversary analysis is complete.')
    except:
      import IPython; IPython.embed(); raise SystemExit
