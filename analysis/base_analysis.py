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
    self.analysis_log_file = self.analysis_out_dir+"analysis.log"
    if "overwrite_analysis" in params.keys() and params["overwrite_analysis"]:
      if os.path.exists(self.analysis_log_file):
        os.remove(self.analysis_log_file)
    if os.path.exists(self.analysis_log_file) and os.stat(self.analysis_log_file).st_size != 0:
      self.analysis_logger = Logger(self.analysis_log_file, overwrite=False)
      analysis_text = self.analysis_logger.load_file()
      prev_analysis_params = self.analysis_logger.read_params(analysis_text)
      if type(prev_analysis_params) == dict: # there was only one param entry
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

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.analysis_out_dir):
      os.makedirs(self.analysis_out_dir)

  def load_model(self):
    """Load model object into analysis object"""
    self.model = mp.get_model(self.model_params["model_type"])

  def get_log_stats(self):
    """Wrapper function for parsing the log statistics"""
    return self.model_logger.read_stats(self.model_log_text)

  def run_analysis(self, images, save_info=""):
    """
    Wrapper function for running all available model analyses
    Log statistics should be consistent across models, but in general it is expected that
    this method will be overwritten for specific models
    """
    self.run_stats = self.get_log_stats()

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

  def compute_atcs(self, activities, images):
    """
    Returns activity triggered covariances
    Outputs:
      atcs [np.ndarray] of shape (num_pixels, num_pixels, num_neurons)
    Inputs:
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    TODO: is it possible to vectorize this so it is faster?
    """
    num_batch, num_neurons = activities.shape
    images -= np.mean(images)
    atas = self.compute_atas(activities, images)
    atcs = [None,]*num_neurons
    for neuron_id in range(num_neurons):
      ata = atas[:, neuron_id] # pixels x 1
      img_deviations = images - ata[None, :] # batch x pixels
      covs = [x.T.dot(x) for x in img_deviations] # pixels x pixels x batch
      covs_acts = zip(covs, activities[:, neuron_id]) # (PxPxB, b)
      atcs[neuron_id] = np.sum([cov*act for cov,act in covs_acts]) / num_batch
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
      phase=phase, contrast=contrast, diameter=diameter)).reshape(num_pixels)
    # Output arrays
    base_max_responses = np.zeros((num_neurons, num_contrasts))
    test_max_responses = np.zeros((num_neurons, num_contrasts, num_contrasts, num_orientations))
    base_mean_responses = np.zeros((num_neurons, num_contrasts))
    test_mean_responses = np.zeros((num_neurons, num_contrasts, num_contrasts, num_orientations))
    base_rect_responses = np.zeros((num_neurons, 2, num_contrasts))
    test_rect_responses = np.zeros((num_neurons, 2, num_contrasts, num_contrasts, num_orientations))
    for bf_idx, neuron_idx in enumerate(neuron_indices): # each neuron produces a base & test output
      for bco_idx, base_contrast in enumerate(contrasts): # loop over base contrast levels
        base_stims = [grating(neuron_idx, base_contrast, base_orientations[bf_idx], base_phase)
          for base_phase in phases]
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
              mask_stims = [grating(neuron_idx, mask_contrast, mask_orientaiton, mask_phase)
                for mask_phase in phases]
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
      "base_max_responses":base_max_responses, "base_mean_responses":base_mean_responses,
      "test_max_responses":test_max_responses, "test_mean_responses":test_mean_responses}

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
    proj_matrix = np.stack([bf1,v], axis=0)
    return proj_matrix
