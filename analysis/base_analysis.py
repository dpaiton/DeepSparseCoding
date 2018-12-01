import os
import numpy as np
from utils.logger import Logger
import utils.plot_functions as pf
import models.model_picker as mp
import utils.data_processing as dp
from data.dataset import Dataset
import tensorflow as tf

class Analyzer(object):
  def __init__(self, params):
    # Load model parameters and schedule
    self.model_log_file = (params["model_dir"]+"/logfiles/"+params["model_name"]
      +"_v"+params["version"]+".log")
    self.model_logger = Logger(self.model_log_file, overwrite=False)
    self.model_log_text = self.model_logger.load_file()
    self.model_params = self.model_logger.read_params(self.model_log_text)
    self.model_params["rand_state"] = np.random.RandomState(self.model_params["rand_seed"])
    self.rand_state = self.model_params["rand_state"]
    self.model_schedule = self.model_logger.read_schedule(self.model_log_text)
    # Load or create analysis params log
    self.analysis_out_dir = params["model_dir"]+"/analysis/"+params["version"]+"/"
    self.make_dirs() # If analysis log does not exist then we want to make the folder first
    self.analysis_log_file = self.analysis_out_dir+"/logfiles/analysis.log"
    if "overwrite_analysis_log" in params.keys() and params["overwrite_analysis_log"]:
      if os.path.exists(self.analysis_log_file):
        os.remove(self.analysis_log_file)
    if os.path.exists(self.analysis_log_file) and os.stat(self.analysis_log_file).st_size != 0:
      # TODO: This code needs to be well tested - I don't think it is always behaving as expected
      self.analysis_logger = Logger(self.analysis_log_file, overwrite=False)
      analysis_text = self.analysis_logger.load_file()
      prev_analysis_params = self.analysis_logger.read_params(analysis_text)
      if type(prev_analysis_params) == dict: # there was only one param entry
        prev_analysis_params.pop("save_info")
        params.update(prev_analysis_params)
      else: # type is list, which means where were multiple param entries in the log
        for param_item in prev_analysis_params:
          params.update(param_item)
    else: # File is empty
      self.analysis_logger = Logger(self.analysis_log_file, overwrite=True)
      self.analysis_logger.log_params(params)
    self.load_params(params)
    self.load_model() # Adds "self.model" member variable that is another model class

  def load_params(self, params):
    """
    Load analysis parameters into object
    TODO: cp_load_step is not utilized.
    """
    self.analysis_params = params
    self.model_name = params["model_name"]
    self.version = params["version"]
    self.cp_loc = tf.train.latest_checkpoint(params["model_dir"]+"/checkpoints/",
      latest_filename="latest_checkpoint_v"+self.version)
    self.model_params["model_out_dir"] = self.analysis_out_dir
    if "device" in params.keys():
      self.device = params["device"]
    else:
      self.device = self.model_params["device"]
    if "data_dir" in params.keys():
      self.model_params["data_dir"] = params["data_dir"]
    if "rand_seed" in params.keys():
      self.rand_seed = params["rand_seed"]
      self.rand_state = np.random.RandomState(self.rand_seed)
    if "input_scale" in params.keys():
      self.input_scale = params["input_scale"]
    else:
      self.input_scale = 1.0
    # BF Fits
    if "do_basis_analysis" in params.keys():
      self.do_basis_analysis = params["do_basis_analysis"]
    else:
      self.do_basis_analysis = False
    if "ft_padding" in params.keys():
      self.ft_padding = params["ft_padding"]
    else:
      self.ft_padding = None
    if "num_gauss_fits" in params.keys():
      self.num_gauss_fits = params["num_gauss_fits"]
    else:
      self.num_gauss_fits = 20
    if "gauss_thresh" in params.keys():
      self.gauss_thresh = params["gauss_thresh"]
    else:
      self.gauss_thresh = 0.2
    # Activity Triggered Averages
    if "do_atas" in params.keys():
      self.do_atas = params["do_atas"]
    else:
      self.do_atas = False
    if "num_noise_images" in params.keys():
      self.num_noise_images = params["num_noise_images"]
    else:
      self.num_noise_images = 100
    # Adversarial analysis
    if "do_adversaries" in params.keys():
      self.do_adversaries = params["do_adversaries"]
      if "adversarial_eps" in params.keys():
        self.adversarial_eps = params["adversarial_eps"]
      else:
        self.adversarial_eps = 0.01
      if "adversarial_num_steps" in params.keys():
        self.adversarial_num_steps = params["adversarial_num_steps"]
      else:
        self.adversarial_num_steps = 200
      if "adversarial_input_id" in params.keys():
        self.adversarial_input_id = params["adversarial_input_id"]
      else:
        self.adversarial_input_id = 0
      if "adversarial_target_id" in params.keys():
        self.adversarial_target_id = params["adversarial_target_id"]
      else:
        self.adversarial_target_id = 1
    else:
      self.do_adversaries = False
    #  Orientation Selectivity
    if "do_orientation_analysis" in params.keys() and params["do_orientation_analysis"]:
      self.do_orientation_analysis = params["do_orientation_analysis"]
      if "neuron_indices" in params.keys():
        self.ot_neurons = params["neuron_indices"]
      else:
        self.ot_neurons = None
      if "contrasts" in params.keys():
        self.ot_contrasts = params["contrasts"]
      else:
        self.ot_contrasts = None
      if "orientations" in params.keys():
        self.ot_orientations = params["orientations"]
      else:
        self.ot_orientations = None
      if "phases" in params.keys():
        self.ot_phases = params["phases"]
      else:
        self.ot_phases = None
    else:
      self.ot_neurons = None
      self.ot_contrasts = None
      self.ot_orientations = None
      self.ot_phases = None

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.analysis_out_dir):
      os.makedirs(self.analysis_out_dir)
    if not os.path.exists(self.analysis_out_dir+"/savefiles"):
      os.makedirs(self.analysis_out_dir+"/savefiles")
    if not os.path.exists(self.analysis_out_dir+"/logfiles"):
      os.makedirs(self.analysis_out_dir+"/logfiles")

  def load_model(self):
    """Load model object into analysis object"""
    self.model = mp.get_model(self.model_params["model_type"])

  def setup_model(self, params, schedule):
    """
    Run model setup, but also add adversarial nodes to graph
    """
    self.model.load_schedule(schedule)
    self.model.sched_idx = 0
    self.model.load_params(params)
    self.model.check_params()
    self.model.make_dirs()
    self.model.init_logging()
    self.model.log_params()
    self.model.log_schedule()
    self.model.graph = tf.Graph()
    self.model.build_graph()
    self.add_pre_init_ops_to_graph()
    self.model.add_optimizers_to_graph()
    self.model.add_initializer_to_graph()
    self.model.construct_savers()

  def add_pre_init_ops_to_graph(self):
    if self.do_adversaries:
      self.add_adversarial_ops_to_graph()

  def get_log_stats(self):
    """Wrapper function for parsing the log statistics"""
    return self.model_logger.read_stats(self.model_log_text)

  def stats_analysis(self, save_info):
    """Run stats extracted from the logfile"""
    run_stats = self.get_log_stats()
    np.savez(self.analysis_out_dir+"savefiles/run_stats_"+save_info+".npz", data={"run_stats":run_stats})
    self.analysis_logger.log_info("Run stats analysis is complete.")
    return run_stats

  def eval_analysis(self, images, var_names, save_info):
    evals = self.evaluate_model(images, var_names)
    np.savez(self.analysis_out_dir+"savefiles/evals_"+save_info+".npz", data={"evals":evals})
    self.analysis_logger.log_info("Image analysis is complete.")
    return evals

  def basis_analysis(self, weights, save_info):
    bf_stats = dp.get_dictionary_stats(weights, padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    np.savez(self.analysis_out_dir+"savefiles/basis_"+save_info+".npz", data={"bf_stats":bf_stats})
    self.analysis_logger.log_info("Dictionary analysis is complete.")
    return bf_stats

  def ata_analysis(self, images, activity, save_info):
    atas = self.compute_atas(activity, images)
    atcs = self.compute_atcs(activity, images, atas)
    np.savez(self.analysis_out_dir+"savefiles/resopnse_"+save_info+".npz",
      data={"atas":atas, "atcs":atcs})
    self.analysis_logger.log_info("Activity triggered analysis is complete.")
    return (atas, atcs)

  def run_noise_analysis(self, save_info, batch_size=100):
    """
    TODO: compute per batch
    """
    noise_shape = [self.num_noise_images] + self.model_params["data_shape"]
    noise_images = self.rand_state.standard_normal(noise_shape)
    noise_activity = self.compute_activations(noise_images)
    noise_atas = self.compute_atas(noise_activity, noise_images)
    noise_atcs = self.compute_atcs(noise_activity, noise_images, noise_atas)
    np.savez(self.analysis_out_dir+"savefiles/noise_responses_"+save_info+".npz",
      data={"num_noise_images":self.num_noise_images, "noise_activity":noise_activity,
      "noise_atas":noise_atas, "noise_atcs":noise_atcs})
    self.analysis_logger.log_info("Noise analysis is complete.")
    return (noise_activity, noise_atas, noise_atcs)

  def grating_analysis(self, weight_stats, save_info):
    ot_grating_responses = self.orientation_tuning(weight_stats, self.ot_contrasts,
      self.ot_orientations, self.ot_phases, self.ot_neurons, scale=self.input_scale)
    np.savez(self.analysis_out_dir+"savefiles/ot_responses_"+save_info+".npz", data=ot_grating_responses)
    ot_mean_activations = ot_grating_responses["mean_responses"]
    base_orientations = [self.ot_orientations[np.argmax(ot_mean_activations[bf_idx,-1,:])]
      for bf_idx in range(len(ot_grating_responses["neuron_indices"]))]
    co_grating_responses = self.cross_orientation_suppression(self.bf_stats,
      self.ot_contrasts, self.ot_phases, base_orientations, self.ot_orientations, self.ot_neurons,
      scale=self.input_scale)
    np.savez(self.analysis_out_dir+"savefiles/co_responses_"+save_info+".npz", data=co_grating_responses)
    self.analysis_logger.log_info("Grating  analysis is complete.")
    return (ot_grating_responses, co_grating_responses)

  def run_patch_recon_analysis(self, full_image, save_info):
    """
    Break image into patches, compute recons, reassemble recons back into a full image
    """
    self.full_image = full_image
    if self.model_params["whiten_data"]:
      # FT method is the only one that works on full images
      wht_img, img_mean, ft_filter = dp.whiten_data(full_image,
        method="FT", lpf_cutoff=self.model_params["lpf_cutoff"])
    else:
      wht_img = full_image
    img_patches = dp.extract_patches(wht_img,
      out_shape=(1, self.model_params["patch_edge_size"], self.model_params["patch_edge_size"], 1),
      overlapping=False, randomize=False, var_thresh=0.0)
    img_patches, orig_shape = dp.reshape_data(img_patches, flatten=True)[:2]
    model_eval = self.evaluate_model(img_patches,
      ["inference/activity:0", "output/reconstruction:0"])
    recon_patches = model_eval["output/reconstruction:0"]
    a_vals = model_eval["inference/activity:0"]
    self.recon_frac_act = np.array(np.count_nonzero(a_vals) / float(a_vals.size))
    recon_patches = dp.reshape_data(recon_patches, flatten=False, out_shape=orig_shape)[0]
    self.full_recon = dp.patches_to_image(recon_patches, full_image.shape).astype(np.float32)
    if self.model_params["whiten_data"]:
      self.full_recon = dp.unwhiten_data(self.full_recon, img_mean, ft_filter, method="FT")
    np.savez(self.analysis_out_dir+"savefiles/full_recon_"+save_info+".npz",
      data={"full_image":self.full_image, "full_recon":self.full_recon,
      "recon_frac_act":self.recon_frac_act})
    self.analysis_logger.log_info("Patch recon analysis is complete.")

  def run_analysis(self, images, save_info=""):
    """
    Wrapper function for running all available model analyses
    Log statistics should be consistent across models, but in general it is expected that
    this method will be overwritten for specific models
    """
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
    bf_file_loc = self.analysis_out_dir+"savefiles/basis_"+save_info+".npz"
    if os.path.exists(bf_file_loc):
      self.bf_stats = np.load(bf_file_loc)["data"].item()["bf_stats"]
    # Activity triggered analysis
    act_file_loc = self.analysis_out_dir+"savefiles/response_"+save_info+".npz"
    if os.path.exists(act_file_loc):
      act_analysis = np.load(act_file_loc)["data"].item()
      self.atas = act_analysis["atas"]
      self.atcs = act_analysis["atcs"]
    noise_file_loc = self.analysis_out_dir+"savefiles/noise_responses_"+save_info+".npz"
    if os.path.exists(noise_file_loc):
      noise_analysis = np.load(noise_file_loc)["data"].item()
      self.noise_activity = noise_analysis["noise_activity"]
      self.noise_atas = noise_analysis["noise_atas"]
      self.noise_atcs = noise_analysis["noise_atcs"]
      self.num_noise_images = self.noise_activity.shape[0]
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
    # Adversarial analysis
    adversarial_file_loc = self.analysis_out_dir+"savefiles/adversary_"+save_info+".npz"
    if os.path.exists(adversarial_file_loc):
      data = np.load(adversarial_file_loc)["data"].item()
      self.adversarial_input_image = data["input_image"]
      self.adversarial_target_image = data["target_image"]
      self.adversarial_images = data["adversarial_images"]
      self.adversarial_recons = data["adversarial_recons"]
      self.adversarial_eps = data["eps"]
      self.adversarial_num_steps = data["num_steps"]
      self.adversarial_input_id = data["input_id"]
      self.adversarial_target_id = data["target_id"]
      self.adversarial_input_target_mses = data["input_target_mse"]
      self.adversarial_input_recon_mses = data["input_recon_mses"]
      self.adversarial_input_adv_mses = data["input_adv_mses"]
      self.adversarial_target_recon_mses = data["target_recon_mses"]
      self.adversarial_target_adv_mses = data["target_adv_mses"]
      self.adversarial_adv_recon_mses = data["adv_recon_mses"]

  def evaluate_model(self, images, var_names):
    """
    Creates a session with the loaded model graph to run all tensors specified by var_names
    Outputs:
      evals [dict] containing keys that match var_names and the values computed from the session run
    Inputs:
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
      var_names [list of str] list of strings containing the tf variable names to be evaluated
    """
    feed_dict = self.model.get_feed_dict(images)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def compute_activations(self, images):
    """
    Computes the output code for a set of images.
    It is assumed that the model has a member variable "model.a" that is evaluated
    Alternatively, one can overwrite this method for individual models
    Outputs:
      evaluated model.a on the input images
    Inputs:
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      activations = sess.run(self.model.a, feed_dict)
    return activations

  def compute_atas(self, activities, images):
    """
    Returns activity triggered averages
    Outputs:
      atas [np.ndarray] of shape (num_pixels, num_neurons)
    Inputs:
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    atas = images.T.dot(activities) / images.shape[0]
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
      params={"whiten_data":self.model_params["whiten_data"],
      "whiten_method":self.model_params["whiten_method"]})
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
          params={"whiten_data":self.model_params["whiten_data"],
          "whiten_method":self.model_params["whiten_method"]})
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
                params={"whiten_data":self.model_params["whiten_data"],
                "whiten_method":self.model_params["whiten_method"]})
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
    if "whiten_data" in self.model_params.keys() and self.model_params["whiten_data"]:
      phase_stims, phase_mean, phase_filter = \
        dp.whiten_data(raw_phase_stims, method=self.model_params["whiten_method"])
    if "lpf_data" in self.model_params.keys() and self.model_params["lpf_data"]:
      phase_stims, phase_mean, phase_filter = \
        dp.lpf_data(raw_phase_stims, cutoff=self.model_params["lpf_cutoff"])
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
            if "whiten_data" in self.model_params.keys() and self.model_params["whiten_data"]:
              test_stims, test_mean, test_filter = \
                dp.whiten_data(raw_test_stims, method=self.model_params["whiten_method"])
            if "lpf_data" in self.model_params.keys() and self.model_params["lpf_data"]:
              test_stims, test_mean, test_filter = \
                dp.lpf_data(raw_test_stims, cutoff=self.model_params["lpf_cutoff"])
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

  def neuron_angles(self, bf_stats):
    """
    Compute the angle between all pairs of basis functions in bf_stats
    Outputs:
      neuron_angles [np.ndarray] of shape [num_neurons, num_neurons] with all angles
    Inputs:
      bf_stats [dict] returned from utils/data_processing.get_dictionary_stats()
    """
    num_pixels = self.model_params["patch_edge_size"]**2
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

  def add_adversarial_ops_to_graph(self):
    """
    Append opes to the graph for adversarial analysis
    """
    with tf.device(self.model.device):
      with self.model.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.model.orig_input = tf.placeholder(tf.float32, shape=self.model.x_shape,
            name="original_input_data")
          self.model.adv_target = tf.placeholder(tf.float32, shape=self.model.x_shape,
            name="adversarial_target_data")
          self.model.recon_mult = tf.placeholder(tf.float32, name="recon_mult")
        with tf.name_scope("loss") as scope:
          # Want to avg over batch, sum over the rest
          reduc_dim = list(range(1, len(self.model.a.shape)))
          self.model.adv_recon_loss = tf.reduce_mean(0.5 *
            tf.reduce_sum(tf.square(tf.subtract(self.model.adv_target,
            self.model.compute_recon(self.model.a))), axis=reduc_dim),
            name="target_recon_loss")
          self.model.input_pert_loss = tf.reduce_mean(0.5 *
            tf.reduce_sum(tf.square(tf.subtract(self.model.orig_input,
            self.model.x)), axis=reduc_dim),
            name="input_perturbed_loss")
          self.model.carlini_loss = tf.add_n([self.model.input_pert_loss, tf.multiply(self.model.recon_mult,
            self.model.adv_recon_loss)])
        with tf.name_scope("adversarial") as scope:
          self.model.fast_sign_adv_dx = -tf.sign(tf.gradients(self.model.adv_recon_loss, self.model.x)[0])
          self.model.carlini_adv_dx = -tf.gradients(self.model.carlini_loss, self.model.x)[0]

  def construct_adversarial_stimulus(self, input_image, target_image, eps=0.01, num_steps=10):
    mse = lambda x,y: np.mean(np.square(x - y))
    losses = []
    adversarial_images = []
    recons = []
    input_recon_mses = []
    input_adv_mses = []
    target_recon_mses = []
    target_adv_mses = []
    adv_recon_mses = []
    input_target_mse = mse(input_image, target_image)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      ## Setup session
      feed_dict = self.model.get_feed_dict(input_image)
      feed_dict[self.model.adv_target] = target_image
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      new_image = input_image.copy()
      for step in range(num_steps):
        adversarial_images.append(new_image.copy())
        self.analysis_logger.log_info("Adversarial analysis, step "+str(step))
        eval_ops = [self.model.x_, self.model.fast_sign_adv_dx]
        recon, fast_sign_adv_dx = sess.run(eval_ops, feed_dict)
        new_image += eps * fast_sign_adv_dx
        input_recon_mses.append(mse(input_image, recon))
        input_adv_mses.append(mse(input_image, new_image))
        target_recon_mses.append(mse(target_image, recon))
        target_adv_mses.append(mse(target_image, new_image))
        adv_recon_mses.append(mse(new_image, recon))
        recons.append(recon)
        feed_dict[self.model.x] = new_image
      mses = {"input_target_mse":input_target_mse, "input_recon_mses":input_recon_mses,
      "input_adv_mses":input_adv_mses, "target_recon_mses":target_recon_mses,
      "target_adv_mses":target_adv_mses, "adv_recon_mses":adv_recon_mses}
      return adversarial_images, recons, mses

  def adversary_analysis(self, images, input_id=0, target_id=1, eps=0.01, num_steps=100,
    save_info=""):
    input_image = images[input_id, ...][None,...].astype(np.float32)
    target_image = images[target_id, ...][None,...].astype(np.float32)
    self.adversarial_images, self.adversarial_recons, mses = self.construct_adversarial_stimulus(input_image,
      target_image, eps, num_steps)
    self.adversarial_input_target_mses = mses["input_target_mse"]
    self.adversarial_input_recon_mses = mses["input_recon_mses"]
    self.adversarial_input_adv_mses = mses["input_adv_mses"]
    self.adversarial_target_recon_mses = mses["target_recon_mses"]
    self.adversarial_target_adv_mses = mses["target_adv_mses"]
    self.adversarial_adv_recon_mses = mses["adv_recon_mses"]
    self.adversarial_input_image = input_image
    self.adversarial_target_image = target_image
    out_dict = {"input_image": input_image, "target_image":target_image,
      "adversarial_images":self.adversarial_images, "adversarial_recons":self.adversarial_recons,
      "eps":eps, "num_steps":num_steps, "input_id":input_id, "target_id":target_id}
    out_dict.update(mses)
    np.savez(self.analysis_out_dir+"savefiles/adversary_"+save_info+".npz", data=out_dict)
    self.analysis_logger.log_info("Adversary analysis is complete.")
    return self.adversarial_images, self.adversarial_recons, mses
