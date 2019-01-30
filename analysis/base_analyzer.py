import os
import numpy as np
from utils.logger import Logger
import utils.plot_functions as pf
import models.model_picker as mp
import utils.data_processing as dp
from data.dataset import Dataset
import tensorflow as tf
import tensorflow_compression as tfc
import pdb

class Analyzer(object):
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
  def setup(self, input_params):
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
    self.model_params.model_out_dir = self.analysis_out_dir
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

  #If build_graph gets called without parameters,
  #build placeholders call build graph with default input
  def build_graph(self):
    #Only need to do this if doing adversarial analysis
    if(self.analysis_params.do_class_adversaries or
      self.analysis_params.do_recon_adversaries):
      with tf.device(self.model.params.device):
        with self.model.graph.as_default():
          #Build model's placeholder
          input_placeholder = self.model.build_input_placeholder()
          input_shape = self.model.get_input_shape()

          #Initialize adversarial image variable with original image
          init_val = input_placeholder

          self.adv_var = tf.Variable(init_val,
            dtype=tf.float32, trainable=True, validate_shape=False)
          #Add this variable to model's full_model_ignore_list
          #Otherwise, load_full_model will try to load this
          self.model.full_model_load_ignore.append(self.adv_var)
          #Here, adv_var has a fully dynamic shape. We reshape it to give the variable
          #a semi-dymaic shape (i.e., only batch dimension unknown)
          reshape_adv_var = tf.reshape(self.adv_var, [-1,] + input_shape[1:])
          #If clipping, use bounds for useful gradients
          if self.analysis_params.adversarial_clip:
            self.adv_image = tfc.upper_bound(tfc.lower_bound(
              reshape_adv_var, self.analysis_params.adversarial_clip_min),
              self.analysis_params.adversarial_clip_max)
          else:
            self.adv_image = reshape_adv_var
      self.model.build_graph_from_input(self.adv_image)
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
    #Call own build graph to call model's build graph
    self.build_graph()
    self.model.load_schedule(params.schedule)
    self.model.sched_idx = 0
    self.model.log_schedule()
    self.model.construct_savers()
    self.add_pre_init_ops_to_graph()
    self.model.add_optimizers_to_graph()
    self.model.add_initializer_to_graph()

  def add_pre_init_ops_to_graph(self):
    if self.analysis_params.do_recon_adversaries:
      self.add_recon_adversarial_ops_to_graph()
    if self.analysis_params.do_class_adversaries:
      self.add_class_adversarial_ops_to_graph()

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
      self.run_stats = np.load(stats_file_loc)["data"].item()["run_stats"]
    # var_names evaluated
    eval_file_loc = self.analysis_out_dir+"savefiles/evals_"+save_info+".npz"
    if os.path.exists(eval_file_loc):
      self.evals = np.load(eval_file_loc)["data"].item()["evals"]
    # Basis function fits
    try:
      self.load_basis_stats(save_info)
    except FileNotFoundError:
      self.analysis_logger.log_info("WARNING: Basis stats file not found")
    # Activity triggered analysis
    ata_file_loc = self.analysis_out_dir+"savefiles/atas_"+save_info+".npz"
    if os.path.exists(ata_file_loc):
      ata_analysis = np.load(ata_file_loc)["data"].item()
      self.atas = ata_analysis["atas"]
      self.atcs = ata_analysis["atcs"]
    ata_noise_file_loc = self.analysis_out_dir+"savefiles/atas_noise_"+save_info+".npz"
    if os.path.exists(ata_noise_file_loc):
      ata_noise_analysis = np.load(ata_noise_file_loc)["data"].item()
      self.noise_atas = ata_noise_analysis["noise_atas"]
      self.noise_atcs = ata_noise_analysis["noise_atcs"]
    act_noise_file_loc = self.analysis_out_dir+"savefiles/noise_response_"+save_info+".npz"
    if os.path.exists(act_noise_file_loc):
      noise_analysis = np.load(act_noise_file_loc)["data"].item()
      self.noise_activity = noise_analysis["noise_activity"]
      self.analysis_params.num_noise_images = self.noise_activity.shape[0]
    # Orientation analysis
    tuning_file_locs = [self.analysis_out_dir+"savefiles/ot_responses_"+save_info+".npz",
      self.analysis_out_dir+"savefiles/co_responses_"+save_info+".npz"]
    if os.path.exists(tuning_file_locs[0]):
      self.ot_grating_responses = np.load(tuning_file_locs[0])["data"].item()
    if os.path.exists(tuning_file_locs[1]):
      self.co_grating_responses = np.load(tuning_file_locs[1])["data"].item()
    recon_file_loc = self.analysis_out_dir+"savefiles/full_recon_"+save_info+".npz"
    if os.path.exists(recon_file_loc):
      recon_analysis = np.load(recon_file_loc)["data"].item()
      self.full_image = recon_analysis["full_image"]
      self.full_recon = recon_analysis["full_recon"]
      self.recon_frac_act = recon_analysis["recon_frac_act"]

    #TODO: Assign unique variable names to recon & adv analysis output attributes
    # Recon Adversarial analysis
    recon_adversarial_file_loc = self.analysis_out_dir+"savefiles/recon_adversary_"+save_info+".npz"
    if os.path.exists(recon_adversarial_file_loc):
      data = np.load(recon_adversarial_file_loc)["data"].item()
      self.recon_adversarial_input_images = data["input_images"]
      self.adversarial_target_images = data["target_images"]
      self.adversarial_images = data["adversarial_images"]
      self.adversarial_recons = data["adversarial_recons"]
      self.analysis_params.adversarial_step_size = data["step_size"]
      self.analysis_params.adversarial_num_steps = data["num_steps"]
      self.analysis_params.adversarial_input_id = data["input_id"]
      self.analysis_params.adversarial_target_id = data["target_id"]
      self.adversarial_input_target_mses = data["input_target_mse"]
      self.adversarial_input_recon_mses = data["input_recon_mses"]
      self.adversarial_input_adv_mses = data["input_adv_mses"]
      self.adversarial_target_recon_mses = data["target_recon_mses"]
      self.adversarial_target_adv_mses = data["target_adv_mses"]
      self.adversarial_adv_recon_mses = data["adv_recon_mses"]
      self.adversarial_target_adv_cos_similarities = data["target_adv_cos_similarities"]

    #Class adversarial analysis
    class_adversarial_file_loc = self.analysis_out_dir+"savefiles/class_adversary_"+save_info+".npz"
    if os.path.exists(class_adversarial_file_loc):
      data = np.load(class_adversarial_file_loc)["data"].item()
      self.class_adversarial_input_images = data["input_images"]
      self.adversarial_input_labels = data["input_labels"]
      self.adversarial_target_labels = data["target_labels"]
      self.adversarial_images = data["adversarial_images"]
      self.adversarial_outputs = data["adversarial_outputs"]
      self.analysis_params.adversarial_step_size = data["step_size"]
      self.analysis_params.adversarial_num_steps = data["num_steps"]
      self.analysis_params.adversarial_input_id = data["input_id"]
      self.adversarial_input_adv_mses = data["input_adv_mses"]
      self.adversarial_target_output_losses = data["target_output_losses"]

  def load_basis_stats(self, save_info):
    bf_file_loc = self.analysis_out_dir+"savefiles/basis_"+save_info+".npz"
    self.bf_stats = np.load(bf_file_loc)["data"].item()["bf_stats"]

  def stats_analysis(self, save_info):
    """Run stats extracted from the logfile"""
    run_stats = self.get_log_stats()
    np.savez(self.analysis_out_dir+"savefiles/run_stats_"+save_info+".npz", data={"run_stats":run_stats})
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

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

  def compute_activations(self, images):
    """
    Computes the output code for a set of images.
    Outputs:
      evaluated model.get_encodings() on the input images
    Inputs:
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images, is_test=True)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      activations = sess.run(self.model.get_encodings(), feed_dict)
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
    phase_stims = self.model.preprocess_dataset(phase_stims,
      params={"whiten_data":self.model_params.whiten_data,
      "whiten_method":self.model_params.whiten_method})
    phase_stims = self.model.reshape_dataset(phase_stims, self.model_params)
    phase_stims["test"].images /= np.max(np.abs(phase_stims["test"].images))
    phase_stims["test"].images *= scale
    activations = self.compute_activations(phase_stims["test"].images).reshape(num_neurons,
      num_contrasts, num_orientations, num_phases, tot_num_bfs)
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

  def neuron_angles(self, bf_stats):
    """
    Compute the angle between all pairs of basis functions in bf_stats
    Outputs:
      neuron_angles [np.ndarray] of shape [num_neurons, num_neurons] with all angles
    Inputs:
      bf_stats [dict] returned from utils/data_processing.get_dictionary_stats()
    """
    num_pixels = self.model_params.patch_edge_size**2
    neuron_angles = np.zeros((bf_stats["num_outputs"], bf_stats["num_outputs"]))
    for neuron1 in range(bf_stats["num_outputs"]):
      for neuron2 in range(bf_stats["num_outputs"]):
        bf1 = bf_stats["basis_functions"][neuron1].reshape((num_pixels,1))
        bf2 = bf_stats["basis_functions"][neuron2].reshape((num_pixels,1))
        inner_products = np.dot((bf1/np.linalg.norm(bf1)).T, bf2/np.linalg.norm(bf2))
        inner_products[inner_products>1.0] = 1.0
        inner_products[inner_products<-1.0] = -1.0
        angle = np.arccos(inner_products)
        neuron_angles[neuron1, neuron2] = angle
    return neuron_angles

  def bf_projections(self, bf1, bf2):
    """
    Find a projection basis that is orthogonal to bf1 and as close as possible to bf2
    Usees a single step of the Gram-Schmidt process
    Outputs:
      projection_matrix [tuple] containing [ax_1, ax_2] for projecting data into the 2d array
    Inputs
      bf1 [np.ndarray] of shape [num_pixels,]
      bf2 [np.ndarray] of shape [num_pixels,]
    """
    v = bf2 - np.dot(bf2[:,None].T, bf1[:,None]) * bf1
    v = np.squeeze((v / np.linalg.norm(v)).T)
    proj_matrix = np.stack([bf1, v], axis=0)
    return proj_matrix, v

  def add_recon_adversarial_ops_to_graph(self):
    """
    Append opes to the graph for adversarial analysis
    """
    with tf.device(self.analysis_params.device):
      with self.model.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.adv_target = tf.placeholder(tf.float32, shape=self.model.get_input_shape(),
            name="adversarial_target_data")
          self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")

        with tf.name_scope("loss") as scope:
          self.recon = self.model.compute_recon_from_encoding(self.model.get_encodings())
          self.adv_recon_loss = 0.5 * tf.reduce_sum(
            tf.square(tf.subtract(self.adv_target, self.recon)),
            name="target_recon_loss")
          if(self.analysis_params.adversarial_attack_method == "carlini"):
            self.input_pert_loss = 0.5 * tf.reduce_sum(
              tf.square(self.model.input_placeholder - self.adv_image),
              name="input_perturbed_loss")
            self.adv_carlini_loss = (1 - self.recon_mult) * self.input_pert_loss \
              + self.recon_mult * self.adv_recon_loss

        with tf.name_scope("optimizer") as scope:
          if(self.analysis_params.adversarial_attack_method == "kurakin"):
            self.adv_grad = -tf.sign(tf.gradients(self.adv_recon_loss, self.adv_var)[0])
            self.adv_update_op = self.adv_var.assign_add(
              self.analysis_params.adversarial_step_size * self.adv_grad)
          elif(self.analysis_params.adversarial_attack_method == "carlini"):
            self.adv_opt = tf.train.AdamOptimizer(
              learning_rate = self.analysis_params.adversarial_step_size)
            #Find gradient wrt self.model.input_variable, but apply them to tmp variable
            self.adv_grads = self.adv_opt.compute_gradients(
              self.adv_carlini_loss, var_list=[self.adv_var])
            self.adv_update_op = self.adv_opt.apply_gradients(self.adv_grads)
          else:
            assert False, ("Adversarial attack method must be \"kurakin\" or \"carlini\"")

  def construct_recon_adversarial_stimulus(self, input_images, target_images,
    step_size=0.01, num_steps=10):

    input_target_mse = dp.mse(input_images, target_images)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if(self.analysis_params.adversarial_attack_method == "kurakin"):
      #Not using recon_mult here, so set arb value
      self.analysis_params.recon_mult = [0]
    elif(self.analysis_params.adversarial_attack_method == "carlini"):
      if(type(self.analysis_params.recon_mult) is not list):
        self.analysis_params.recon_mult = [self.analysis_params.recon_mult]
    else:
      assert False, ("Adversarial attack method must be \"kurakin\" or \"carlini\"")

    distances = {"input_target_mse":[], "input_recon_mses":[],
    "input_adv_mses":[], "target_recon_mses":[],
    "target_adv_mses":[], "adv_recon_mses":[], "target_adv_cos_similarities":[]}
    all_adversarial_images = []
    all_recons = []
    for r_mult in self.analysis_params.recon_mult:
      adversarial_images = []
      recons = []
      input_recon_mses = []
      input_adv_mses = []
      target_recon_mses = []
      target_adv_mses = []
      adv_recon_mses = []
      target_adv_angles = []

      with tf.Session(config=config, graph=self.model.graph) as sess:
        feed_dict = self.model.get_feed_dict(input_images, is_test=True)
        feed_dict[self.model.input_placeholder] = input_images
        feed_dict[self.adv_target] = target_images
        feed_dict[self.recon_mult] = r_mult
        sess.run(self.model.init_op, feed_dict)
        self.model.load_full_model(sess, self.analysis_params.cp_loc)
        for step in range(num_steps):
          #Stats
          if(step%self.analysis_params.adversarial_save_int == 0):
            recon_eval, adv_image_eval = sess.run([self.recon, self.adv_image], feed_dict)
            adversarial_images.append(adv_image_eval)
            recons.append(recon_eval)
            input_recon_mses.append(dp.mse(input_images, recon_eval))
            input_adv_mses.append(dp.mse(input_images, adv_image_eval))
            target_recon_mses.append(dp.mse(target_images, recon_eval))
            target_adv_mses.append(dp.mse(target_images, adv_image_eval))
            adv_recon_mses.append(dp.mse(adv_image_eval, recon_eval))
            target_adv_angles.append(dp.cos_similarity(input_images, adv_image_eval))

          self.analysis_logger.log_info("Recon Adversarial analysis, step "+str(step))
          #Run update op
          sess.run(self.adv_update_op, feed_dict)

      distances["input_target_mse"].append(input_target_mse)
      distances["input_recon_mses"].append(input_recon_mses)
      distances["input_adv_mses"].append(input_adv_mses)
      distances["target_recon_mses"].append(target_recon_mses)
      distances["target_adv_mses"].append(target_adv_mses)
      distances["adv_recon_mses"].append(adv_recon_mses)
      distances["target_adv_cos_similarities"].append(target_adv_angles)
      all_adversarial_images.append(adversarial_images)
      all_recons.append(recons)
    return all_adversarial_images, all_recons, distances

  def recon_adversary_analysis(self, images, labels=None, batch_size=1, input_id=None,
    target_method="random", target_id=None, step_size=0.01, num_steps=100, save_info=""):

    #Default parameters
    if input_id is None:
      input_id = np.arange(batch_size).astype(np.int32)
    else:
      input_id = np.array(input_id)

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
        while(np.any(target_id == input_id) or has_same_labels):
          resample_idx = np.nonzero(target_id == input_id)
          target_id[resample_idx] = self.rand_state.randint(
            0, num_images, size=resample_idx[0].shape)
    elif(target_method == "specified"):
      assert(target_id is not None)
      target_id = np.array(target_id)
      assert(target_id.shape[0] == batch_size)
    else:
      assert False, ("Allowed target methods for recon adversary are " +
        "\"random\", \"random_different_labels\", or \"specified\"")

    #Define target label based on target method
    target_images = images[target_id, ...].astype(np.float32)
    stims = self.construct_recon_adversarial_stimulus(input_images,
      target_images, step_size, num_steps)
    self.adversarial_images, self.adversarial_recons, distances = stims

    self.adversarial_input_target_mses = distances["input_target_mse"]
    self.adversarial_input_recon_mses = distances["input_recon_mses"]
    self.adversarial_input_adv_mses = distances["input_adv_mses"]
    self.adversarial_target_recon_mses = distances["target_recon_mses"]
    self.adversarial_target_adv_mses = distances["target_adv_mses"]
    self.adversarial_adv_recon_mses = distances["adv_recon_mses"]
    self.recon_adversarial_input_images = input_images
    self.adversarial_target_images = target_images
    out_dict = {"input_images": input_images, "target_images":target_images,
      "adversarial_images":self.adversarial_images, "adversarial_recons":self.adversarial_recons,
      "step_size":step_size, "num_steps":num_steps, "input_id":input_id, "target_id":target_id}
    out_dict.update(distances)
    np.savez(self.analysis_out_dir+"savefiles/recon_adversary_"+save_info+".npz", data=out_dict)
    self.analysis_logger.log_info("Adversary analysis is complete.")
    return self.adversarial_images, self.adversarial_recons, distances

  def add_class_adversarial_ops_to_graph(self):
    """
    Append opes to the graph for adversarial analysis
    """
    with tf.device(self.analysis_params.device):
      with self.model.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.adv_target = tf.placeholder(tf.float32, shape=self.model.label_shape,
            name="adversarial_target")
          self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")

        with tf.name_scope("loss") as scope:
          if(self.analysis_params.adversarial_attack_method == "kurakin"):
            if(self.analysis_params.adversarial_target_method == "untargeted"):
              #No target attack
              self.adv_loss = tf.negative(self.model.mlp_module.mean_loss)
            else:
              #Targeted attack
              self.adv_loss = -tf.reduce_sum(tf.multiply(self.model.label_est,
                tf.log(tf.clip_by_value(self.adv_target, self.model_params.eps, 1.0))))
          elif(self.analysis_params.adversarial_attack_method == "carlini"):
            self.input_pert_loss = 0.5 * tf.reduce_sum(
              tf.square(tf.subtract(self.model.input_placeholder, self.adv_image)),
              name="input_perturbed_loss")
            #Using method f_6 in carlini et al. 2017 seciton 5, A
            #relu( max_{i != t}(Z(x)_i) - Z(x)_t )
            self.model_logits = self.model.get_encodings()

            #Assuming adv_target is one hot
            with tf.control_dependencies([
              tf.assert_equal(tf.reduce_sum(self.adv_target, axis=-1), 1.0)]):
              self.adv_target = self.adv_target

            #Construct two boolean masks, one with only target class as true
            #and one with everything not target class
            self.target_mask = self.adv_target > .5
            self.not_target_mask = self.adv_target < .5

            #Z(x)_t
            #boolean_mask returns a flattened array, so need to reshape back
            self.logits_target_val = tf.boolean_mask(self.model_logits, self.target_mask)[:, None]
            #max_{i!=t} Z(x)_i
            self.logits_not_target_val = tf.boolean_mask(self.model_logits, self.not_target_mask)
            self.logits_not_target_val = tf.reshape(self.logits_not_target_val,
              [-1, self.model.label_shape[-1]-1])

            self.max_logits_not_target_val = tf.reduce_max(self.logits_not_target_val, axis=-1)

            self.target_class_loss = tf.reduce_sum(tf.nn.relu(
              self.max_logits_not_target_val - self.logits_target_val))

            self.adv_loss = self.input_pert_loss + \
              self.recon_mult * self.target_class_loss
          else:
            assert False, ("Adversarial attack method must be \"kurakin\" or \"carlini\"")
          self.output = self.model.label_est

        with tf.name_scope("optimizer") as scope:
          if(self.analysis_params.adversarial_attack_method == "kurakin"):
            self.adv_grad = -tf.sign(tf.gradients(self.adv_loss, self.adv_var)[0])
            self.adv_update_op = self.adv_var.assign_add(
              self.analysis_params.adversarial_step_size * self.adv_grad)
          elif(self.analysis_params.adversarial_attack_method == "carlini"):
            self.adv_opt = tf.train.AdamOptimizer(
              learning_rate = self.analysis_params.adversarial_step_size)
            #Find gradient wrt self.model.x, but apply them to tmp variable
            self.adv_grads = self.adv_opt.compute_gradients(
              self.adv_loss, var_list=[self.adv_var])
            self.adv_update_op = self.adv_opt.apply_gradients(self.adv_grads)

  def construct_class_adversarial_stimulus(self, input_images, input_labels, target_labels,
    step_size=0.01, num_steps=10):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if(self.analysis_params.adversarial_attack_method == "kurakin"):
      #Not using recon_mult here, so set arb value
      self.analysis_params.recon_mult = [0]
    elif(self.analysis_params.adversarial_attack_method == "carlini"):
      assert(target_labels is not None)
      if(type(self.analysis_params.recon_mult) is not list):
        self.analysis_params.recon_mult = [self.analysis_params.recon_mult]
    else:
      assert False, ("Adversarial attack method must be \"kurakin\" or \"carlini\"")

    mses = {"input_adv_mses":[], "target_output_losses":[],}
    all_adversarial_images = []
    all_outputs = []
    for r_mult in self.analysis_params.recon_mult:
      adversarial_images = []
      outputs = []
      input_adv_mses = []
      target_output_losses = []

      with tf.Session(config=config, graph=self.model.graph) as sess:
        feed_dict = self.model.get_feed_dict(input_images, input_labels, is_test=True)
        feed_dict[self.model.input_placeholder] = input_images
        if(target_labels is not None):
          feed_dict[self.adv_target] = target_labels
        feed_dict[self.recon_mult] = r_mult
        sess.run(self.model.init_op, feed_dict)
        self.model.load_full_model(sess, self.analysis_params.cp_loc)
        for step in range(num_steps):
          #Stats
          if(step%self.analysis_params.adversarial_save_int == 0):
            adv_image_eval, output, target_output_loss = \
              sess.run([self.adv_image, self.output, self.adv_loss], feed_dict)
            adversarial_images.append(adv_image_eval)
            outputs.append(output)
            input_adv_mses.append(dp.mse(input_images, adv_image_eval))
            target_output_losses.append(target_output_loss)

          self.analysis_logger.log_info("Class Adversarial analysis, step "+str(step))
          #Run update op
          sess.run(self.adv_update_op, feed_dict)

      mses["input_adv_mses"].append(input_adv_mses)
      mses["target_output_losses"].append(target_output_losses)
      all_adversarial_images.append(adversarial_images)
      all_outputs.append(outputs)
    return all_adversarial_images, all_outputs, mses

  def class_adversary_analysis(self, images, labels, batch_size=1, input_id=None,
      target_method="untargeted", target_labels=None, step_size=0.01, num_steps=100, save_info=""):

    #Default parameters
    #TODO need to make sure these are getting correct classifications
    if input_id is None:
      input_id = np.arange(batch_size).astype(np.int32)
    else:
      input_id = np.array(input_id)

    input_images = images[input_id, ...].astype(np.float32)
    input_labels = labels[input_id, ...].astype(np.float32)
    num_classes = input_labels.shape[-1]

    #Define target label based on target method
    if(target_method == "random"):
      input_classes = np.argmax(input_labels, axis=-1)
      target_labels = input_classes.copy()
      #Resample until target label is not a true label
      while(np.any(target_labels == input_classes)):
        resample_idx = np.nonzero(target_labels == input_classes)
        target_labels[resample_idx] = self.rand_state.randint(0, num_classes, size=resample_idx[0].shape)
    elif(target_method == "specified"):
      assert(target_labels is not None)
      target_labels = np.array(target_labels)
      assert(target_labels.shape[0] == batch_size)
    elif(target_method == "untargeted"):
      target_labels = None
    else:
      assert False, ("Allowed target methods forclassification adversary are " +
        "\"random\", \"untargeted\", or \"specified\"")

    #Check if target_labels is a class or one hot
    #If class, convert to one hot
    if(target_labels is not None):
      if(target_labels.ndim == 1):
        out = np.zeros((batch_size, num_classes))
        #Convert target_labels into idx
        target_labels_idx = (np.arange(batch_size).astype(np.int32), target_labels)
        out[target_labels_idx] = 1
        target_labels = out

    self.adversarial_images, self.adversarial_outputs, mses =  \
      self.construct_class_adversarial_stimulus(input_images, input_labels,
      target_labels, step_size, num_steps)

    self.adversarial_input_adv_mses = mses["input_adv_mses"]
    self.adversarial_target_output_losses = mses["target_output_losses"]
    self.class_adversarial_input_images = input_images
    self.adversarial_input_labels = input_labels
    self.adversarial_target_labels = target_labels

    out_dict = {"input_images": input_images, "input_labels":input_labels, "target_labels":target_labels,
      "adversarial_images":self.adversarial_images, "adversarial_outputs":self.adversarial_outputs,
      "step_size":step_size, "num_steps":num_steps, "input_id":input_id}
    out_dict.update(mses)
    np.savez(self.analysis_out_dir+"savefiles/class_adversary_"+save_info+".npz", data=out_dict)
    self.analysis_logger.log_info("Adversary analysis is complete.")
    return self.adversarial_images, self.adversarial_outputs, mses
