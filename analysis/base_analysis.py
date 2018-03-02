import os
import numpy as np
from scipy.optimize import curve_fit
import utils.log_parser as lp
import utils.plot_functions as pf
import models.model_picker as mp
import utils.data_processing as dp
import tensorflow as tf

class Analyzer(object):
  def __init__(self, params):
    self.log_file = (params["model_dir"]+"/logfiles/"+params["model_name"]
      +"_v"+params["version"]+".log")
    self.log_text = lp.load_file(self.log_file)
    self.model_params = lp.read_params(self.log_text)
    assert self.model_params["model_type"] == params["model_type"], (
      "Model type defined in log text must match model type given in params.")
    self.model_params["rand_state"] = np.random.RandomState(
      self.model_params["rand_seed"])
    self.model_schedule = lp.read_schedule(self.log_text)
    self.load_params(params)
    self.make_dirs()
    self.load_model()
    self.model.log_params(params)

  def load_params(self, params):
    """Load analysis parameters into object"""
    # Model details
    self.model_name = params["model_name"]
    self.version = params["version"]
    self.device = params["device"]
    self.analysis_out_dir = params["model_dir"]+"/analysis/"+self.version+"/"
    if "cp_load_step" in params.keys() and params["cp_load_step"] is not None:
      self.cp_load_step = params["cp_load_step"]
      self.cp_loc = (params["model_dir"]+"/checkpoints/"+params["model_name"]
        +"_v"+params["version"]+"_full-"+str(self.cp_load_step))
    else:
      self.cp_load_step = None
      # TODO: tf.train.latest_checkpoint ends up looking for the directory where the checkpoint was created
      #   not sure where this is set, but it results in a poorly described exit
      self.cp_loc = tf.train.latest_checkpoint(params["model_dir"]+"/checkpoints/")
      #self.cp_loc = params["model_dir"]+"/checkpoints/lca_256_l0_0.5_v2.0_weights-35000"
    self.model_params["model_out_dir"] = self.analysis_out_dir
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
    return lp.read_stats(self.log_text)

  def evaluate_model(self, images, var_names):
    feed_dict = self.model.get_feed_dict(images)
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def compute_activations(self, images):
    with tf.Session(graph=self.model.graph) as sess:
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
    atas = np.dot(images.T, activities)
    avg_atas = atas / images.shape[1]
    return avg_atas

  ## TODO: These use bf_stats, which is created by child methods
  def cross_orientation_suppression(self, contrasts, phases, base_orientations,
    mask_orientations, num_bfs=None, diameter=-1):

    num_bfs = self.bf_stats["num_outputs"] if num_bfs is None else num_bfs
    num_pixels = self.bf_stats["patch_edge_size"]**2

    assert np.asarray(base_orientations).size == num_bfs, (
      "You must specify a base orientation for all basis functions")

    # Generate a grating with spatial frequency estimated from bf_stats
    grating = lambda bf_idx,orientation,phase,contrast:dp.generate_grating(
      *dp.get_grating_params(bf_stats=self.bf_stats, bf_idx=bf_idx, orientation=orientation,
      phase=phase, contrast=contrast, diameter=diameter)).reshape(num_pixels)

    # Stimulus parameters
    num_contrasts = np.asarray(contrasts).size
    num_phases = np.asarray(phases).size
    num_orientations = np.asarray(mask_orientations).size

    #Setup stimulus
    #base_stims = np.zeros((num_bfs*num_contrasts*num_phases, num_pixels))
    #test_stims = np.zeros((num_bfs*num_contrasts*num_contrasts*num_orientations*num_phases*num_phases, num_pixels))
    #test_idx = 0
    #base_idx = 0 
    #for bf_idx in range(num_bfs):
    #  for bco_idx, base_contrast in enumerate(contrasts):
    #    for bph_idx, base_phase in enumerate(phases):
    #      base_stim = grating(bf_idx, base_orientations[bf_idx], base_phase, base_contrast)
    #      base_stims[base_idx, :] = base_stim
    #      base_idx += 1
    #      for co_idx, contrast in enumerate(contrasts):
    #        for or_idx, orientation in enumerate(mask_orientations):
    #          for ph_idx, phase in enumerate(phases):
    #            test_stim = grating(bf_idx, orientation, phase, contrast)
    #            test_stims[test_idx, :] = 0.5*(base_stim + test_stim)
    #            test_idx += 1

    # Each bf has its own stimulus with a set spatial frequency, but we still compute activations
    # for all neurons and all stimuli for batching reasons
    #base_activations = self.compute_activations(base_stims).reshape(num_bfs, num_contrasts, num_phases, num_bfs)
    #test_activations = self.compute_activations(test_stims).reshape(num_bfs, num_contrasts,
    #  num_contrasts, num_orientations, num_phases, num_phases, num_bfs)
    #base_stims = base_stims.reshape(num_bfs, num_contrasts, num_phases, num_pixels)
    #test_stims = test_stims.reshape(num_bfs, num_contrasts, num_contrasts,
    #  num_orientations, num_phases, num_phases, num_pixels)
    
    # Output arrays
    base_responses = np.zeros((num_bfs, num_contrasts))
    test_responses = np.zeros((num_bfs, num_contrasts, num_contrasts, num_orientations))

    for bf_idx in range(num_bfs): # each neuron produces a baseline output & test output
      for bco_idx, base_contrast in enumerate(contrasts): # loop over base contrast levels
        base_stims = np.zeros((num_phases, num_pixels))
        for bph_idx, base_phase in enumerate(phases):
          base_stims[bph_idx] = grating(bf_idx, base_orientations[bf_idx], base_phase, base_contrast)
        base_activity = self.compute_activations(base_stims).reshape(num_phases, num_bfs)[:, bf_idx]
        base_responses[bf_idx, bco_idx] = np.max(base_activity) - np.min(base_activity) # peak-to-trough amplitude
        for co_idx, test_contrast in enumerate(contrasts): # for each base contrast, loop over test contrasts
            test_stims = np.zeros((num_orientations, num_phases**2, num_pixels))
            for or_idx, test_orientation in enumerate(mask_orientations): # loop over test orientations
              mask_stims = np.zeros((num_phases, num_pixels))
              for ph_idx, test_phase in enumerate(phases):
                mask_stims[ph_idx,:] = grating(bf_idx, test_orientation, test_phase, test_contrast)
              test_stims[or_idx,:,:] = 0.5*(base_stims[:,None,:] + mask_stims[None,:,:]).reshape(num_phases**2,
                num_pixels)
            test_stims = test_stims.reshape(num_orientations*num_phases*num_phases, num_pixels)
            test_activity = self.compute_activations(test_stims)[:, bf_idx].reshape(num_orientations, num_phases**2)
            # peak-to-trough amplitude is computed across all base & mask phases
            test_responses[bf_idx, bco_idx, co_idx, :] = np.max(test_activity, axis=1) - np.min(test_activity, axis=1)
    return {"contrasts":contrasts, "phases":phases, "base_orientations":base_orientations,
      "mask_orientations":mask_orientations, "base_responses":base_responses, "test_responses":test_responses}

  def orientation_tuning(self, contrasts, orientations, phases, num_pixels=None,
    num_bfs=None, diameter=-1):
    num_pixels = self.bf_stats["patch_edge_size"]**2 if num_pixels is None else num_pixels
    num_bfs = self.bf_stats["num_outputs"] if num_bfs is None else num_bfs

    # Generate a grating with spatial frequency estimated from bf_stats
    grating = lambda bf_idx,orientation,phase,contrast:dp.generate_grating(
      *dp.get_grating_params(self.bf_stats, bf_idx, orientation=orientation,
      phase=phase, contrast=contrast, diameter=diameter))

    # Stimulus parameters
    num_contrasts = np.asarray(contrasts).size
    num_orientations = np.asarray(orientations).size
    num_phases = np.asarray(phases).size

    # Output arrays
    contrast_activations = np.zeros((num_bfs, num_contrasts, num_orientations))
    phase_activations = np.zeros((num_bfs, num_contrasts, num_orientations, num_phases))
    best_phases = np.zeros((num_bfs, num_contrasts, num_orientations))
    phase_stims = np.stack([grating(bf_idx, orientation, phase, contrast).reshape(num_pixels)
      for bf_idx in range(num_bfs)
      for contrast in contrasts
      for orientation in orientations
      for phase in phases], axis=0)
    # Each bf has its own stimulus with a set spatial frequency, but we still compute activations
    # for all neurons and all stimuli
    activations = self.compute_activations(phase_stims).reshape(num_bfs, num_contrasts,
      num_orientations, num_phases, num_bfs)
    for bf_idx in range(num_bfs):
      for co_idx, contrast in enumerate(contrasts):
        orientation_activations = np.zeros(num_orientations)
        for or_idx, orientation in enumerate(orientations):
          phase_activity = activations[bf_idx, co_idx, or_idx, :, bf_idx]
          best_phases[bf_idx, co_idx, or_idx] = phases[np.argmax(phase_activity)]
          # Choose the peak-to-trough amplitude as the representative output
          orientation_activations[or_idx] = np.max(phase_activity) - np.min(phase_activity)
        contrast_activations[bf_idx, co_idx, :] = orientation_activations
    return {"contrasts":contrasts, "orientations":orientations, "phases":phases,
      "contrast_activations":contrast_activations, "phase_activations":phase_activations,
      "best_phases": best_phases, "phase_stims":phase_stims}
